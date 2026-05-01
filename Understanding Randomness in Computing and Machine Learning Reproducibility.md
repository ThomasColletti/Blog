# Understanding Randomness in Computing and Machine Learning Reproducibility

By Manus AI

Randomness is a fundamental concept in computer science, underpinning everything from secure cryptography to stochastic processes in machine learning. However, the deterministic nature of computers makes generating true randomness a complex challenge. Furthermore, when applying randomness in machine learning (ML), practitioners often encounter a frustrating phenomenon: running the exact same code with the same random seeds inside different Docker images can yield different modeling results, predictions, and learnings. 

This document explores how computers generate randomness and delves into the root causes of non-reproducibility in machine learning across different environments.

## How Computers Generate Randomness

Computers are inherently deterministic machines; given the same inputs and instructions, they will produce the exact same outputs. To generate randomness, systems rely on two primary approaches: True Random Number Generators (TRNGs) and Pseudo-Random Number Generators (PRNGs).

### True Random Number Generators (TRNGs)

TRNGs extract randomness from physical, unpredictable phenomena in the surrounding environment, often referred to as **entropy** [1]. Entropy is a measure of uncertainty or disorder. 

In computing systems, entropy is gathered from various hardware events:
*   **User Interactions:** The exact timing of keystrokes or the micro-movements of a mouse pointer.
*   **System Events:** Interrupt timings, disk I/O operations, and network packet arrival times.
*   **Dedicated Hardware:** Specialized circuits that measure thermal noise, clock jitter, or quantum phenomena (e.g., Intel's Digital Random Number Generator using the `RDRAND` instruction) [1].

The operating system kernel collects these unpredictable bits into an **entropy pool**. For example, in Linux, this pool is continuously updated and mixed using cryptographic hash functions to remove statistical biases [1]. When high-quality randomness is required (such as for generating cryptographic keys), the system reads from this pool (e.g., via `/dev/random`). However, TRNGs are generally slow because they must wait for physical events to occur to replenish the entropy pool.

### Pseudo-Random Number Generators (PRNGs)

Because TRNGs are too slow for applications requiring millions of random numbers per second (like simulations or machine learning), computers use PRNGs. 

A PRNG is a deterministic mathematical algorithm that produces a sequence of numbers that *appears* random. It requires an initial starting value called a **seed**. If a PRNG is initialized with the same seed, it will always produce the exact same sequence of numbers [2].

To ensure the sequence is unpredictable to an outside observer, modern operating systems use Cryptographically Secure Pseudo-Random Number Generators (CSPRNGs). These systems take a small amount of true entropy from the TRNG to use as the seed, and then use fast cryptographic algorithms (like AES or SHA) to generate a vast stream of random bits (e.g., via `/dev/urandom` in Linux) [1].

| Feature | True Random Number Generator (TRNG) | Pseudo-Random Number Generator (PRNG) |
| :--- | :--- | :--- |
| **Source** | Physical phenomena (entropy) | Mathematical algorithms |
| **Determinism** | Non-deterministic | Deterministic (based on seed) |
| **Speed** | Slow (waits for entropy) | Very fast |
| **Primary Use** | Cryptography, seeding PRNGs | Simulations, Machine Learning, Gaming |

## Why Different Docker Images Yield Different ML Results

In machine learning, practitioners often set random seeds (e.g., `np.random.seed(42)`, `torch.manual_seed(42)`) expecting that their model training and inference will be perfectly reproducible. The assumption is that Docker containers encapsulate the environment, ensuring the "build once, run anywhere" promise. 

However, running the same code with the same seeds in different Docker images often results in divergent model weights, loss curves, and final predictions. This occurs because ML reproducibility depends on much more than just the random seed.

### 1. Floating-Point Non-Associativity

The fundamental "original sin" of non-determinism in modern computing is the non-associative nature of floating-point arithmetic [3]. 

Computers represent real numbers using floating-point formats (like FP32 or FP16), which have a fixed number of significant digits. When adding numbers of vastly different magnitudes, the smaller numbers can be rounded off or lost entirely. Consequently, the mathematical rule of associativity `(A + B) + C = A + (B + C)` does not hold true in floating-point math [3].

> "In a toy example, summing a small set of positive and negative values in different orders yields 102 distinct sums due purely to rounding order." [3]

If an algorithm adds a sequence of numbers in a different order, the final sum will be slightly different. In deep learning, where models perform billions of additions and multiplications, these microscopic differences compound rapidly, leading to entirely different model weights and predictions [4].

### 2. Parallelism and Reduction Orders

Modern ML relies heavily on parallel processing via GPUs. When operations like matrix multiplications or reductions (e.g., summing over a dimension in LayerNorm or Attention) are parallelized, the order in which the parallel threads complete and accumulate their results can vary [3].

If a Docker image uses a different version of a deep learning framework (like PyTorch or TensorFlow) or a different underlying CUDA toolkit, the exact parallelization strategy—how the matrices are tiled and distributed across GPU cores—may change. This changes the order of floating-point additions, triggering the non-associativity issue and altering the results [3].

### 3. Underlying Math Library Discrepancies

Deep learning frameworks and libraries like NumPy do not perform matrix math themselves; they delegate these operations to highly optimized Basic Linear Algebra Subprograms (BLAS) libraries. 

Different Docker base images (e.g., Ubuntu vs. Alpine) or different Python package installations might link against different BLAS implementations:
*   **OpenBLAS:** An open-source implementation.
*   **Intel MKL (Math Kernel Library):** Highly optimized for Intel CPUs.
*   **cuBLAS:** NVIDIA's library for GPUs.

Even if the Python code is identical, an `ubuntu:20.04` Docker image might default to a different BLAS library or a different version of OpenBLAS compared to an `ubuntu:22.04` image. Different BLAS libraries use different algorithmic approaches and blocking strategies for matrix multiplication, leading to different floating-point rounding orders and, consequently, different results [5].

### 4. System-Level Dependencies (glibc)

Docker images package the operating system user-space, including the GNU C Library (`glibc`). `glibc` provides the core mathematical functions (like `sin()`, `cos()`, `exp()`) used by higher-level libraries. 

Different OS base images (e.g., Debian 10 vs. Debian 12) contain different versions of `glibc`. Over time, `glibc` maintainers optimize these math functions for speed or accuracy. Therefore, calculating `exp(x)` on `glibc 2.31` might yield a result that differs by a single bit at the end of the mantissa compared to `glibc 2.35`. In the context of a deep neural network, this single-bit difference propagates through activation functions and gradients, eventually causing the model's learning trajectory to diverge.

### 5. Hardware and Driver Variations

While Docker encapsulates the software environment, it does not encapsulate the hardware. When using GPUs, the Docker container relies on the host machine's NVIDIA driver. If the same Docker image is run on a machine with an RTX 3090 and another with an RTX 4080, the underlying hardware architecture (Ampere vs. Ada Lovelace) differs [4]. 

Different GPU architectures have different numbers of Streaming Multiprocessors (SMs) and different Tensor Core behaviors. The cuBLAS library will dynamically select different algorithmic kernels at runtime based on the specific hardware it detects. This dynamic selection changes the execution order, breaking determinism across different hardware, even if the Docker image is identical [4].

## Conclusion

Generating randomness in computing relies on a delicate interplay between unpredictable physical entropy (TRNGs) and fast, deterministic mathematical algorithms (PRNGs). 

In machine learning, the expectation that fixing a PRNG seed guarantees reproducibility is a misconception. When moving between different Docker images, variations in OS-level libraries (`glibc`), math backends (BLAS), framework versions, and parallel reduction strategies alter the exact sequence of floating-point operations. Due to floating-point non-associativity, these microscopic changes compound over billions of operations, leading to divergent modeling results, predictions, and learnings. Achieving true reproducibility requires strict pinning of all software layers—from the OS base image up to the Python packages—and often necessitates running on identical hardware architectures.

---

### References

[1] Red Hat. "Understanding random number generators, and their limitations, in Linux." https://www.redhat.com/en/blog/understanding-random-number-generators-and-their-limitations-linux

[2] Stack Exchange. "TRNG vs PRNG - Entropy?" https://crypto.stackexchange.com/questions/26853/trng-vs-prng-entropy

[3] Thinking Machines Lab. "Defeating Nondeterminism in LLM Inference." https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

[4] Ingonyama. "Solving Reproducibility Challenges in Deep Learning and LLMs: Our Journey." https://www.ingonyama.com/post/solving-reproducibility-challenges-in-deep-learning-and-llms-our-journey

[5] PyTorch Forums. "PyTorch VERY different results on different machines using docker and CPU." https://discuss.pytorch.org/t/pytorch-very-different-results-on-different-machines-using-docker-and-cpu/98590
