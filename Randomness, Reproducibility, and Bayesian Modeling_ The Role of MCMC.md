# Randomness, Reproducibility, and Bayesian Modeling: The Role of MCMC

By Tao Feng

Building upon our previous discussion of how computers generate randomness and why Docker environments can yield non-reproducible machine learning results, we now turn to a domain where randomness is not just a utility, but the core engine of inference: **Bayesian modeling** and **Markov Chain Monte Carlo (MCMC)** sampling.

In traditional deep learning, randomness is primarily used for initialization (e.g., random weights) and stochastic optimization (e.g., shuffling batches in SGD). In Bayesian modeling, however, randomness is used to explore and map out entire high-dimensional probability distributions. This heavy reliance on sequential random sampling makes MCMC algorithms uniquely sensitive to the environmental and computational variations discussed previously.

## Bayesian Modeling and the Intractable Posterior

The goal of Bayesian inference is to update our beliefs about a model's parameters given observed data. This is formalized by Bayes' Theorem:

$$ P(\theta | X) = \frac{P(X | \theta) P(\theta)}{P(X)} $$

Where:
*   **$P(\theta)$** is the **Prior**: Our belief about the parameters $\theta$ before seeing data.
*   **$P(X | \theta)$** is the **Likelihood**: The probability of the data $X$ given the parameters.
*   **$P(\theta | X)$** is the **Posterior**: Our updated belief about the parameters after seeing the data.
*   **$P(X)$** is the **Marginal Likelihood** (or Evidence): The probability of the data averaged over all possible parameter values.

The central challenge in Bayesian modeling is the denominator, $P(X)$. Calculating it requires integrating the likelihood over the entire parameter space: $P(X) = \int P(X | \theta) P(\theta) d\theta$. For complex models with many parameters (like Bayesian Neural Networks or hierarchical models), this high-dimensional integral is analytically impossible to solve and computationally intractable to approximate using standard numerical grids [1].

Because we cannot compute the exact posterior distribution, we must instead draw representative samples from it. This is where MCMC comes in.

## How MCMC Works: Sampling via Randomness

Markov Chain Monte Carlo (MCMC) is a class of algorithms designed to sample from a probability distribution without needing to know its exact normalization constant (the intractable $P(X)$) [2]. 

MCMC works by constructing a "Markov Chain"—a sequence of random variables where the next state depends only on the current state. The algorithm is designed so that, after running for a sufficient amount of time (the "burn-in" period), the chain reaches a stationary distribution that exactly matches the target posterior distribution [2].

### The Role of Randomness in MCMC

Randomness is the driving force of MCMC. Consider the classic **Metropolis-Hastings** algorithm:
1.  **Current State:** The chain is at a specific parameter value, $\theta_{current}$.
2.  **Random Proposal:** The algorithm uses a Pseudo-Random Number Generator (PRNG) to draw a proposed new state, $\theta_{proposed}$, from a proposal distribution (often a Gaussian centered on $\theta_{current}$) [2].
3.  **Acceptance/Rejection:** The algorithm calculates an acceptance ratio based on the likelihood and prior of the proposed state versus the current state. It then draws a random uniform number $u \in [0, 1]$. If the acceptance ratio is greater than $u$, the chain moves to $\theta_{proposed}$; otherwise, it stays at $\theta_{current}$ [2].

Modern probabilistic programming languages (like PyMC or Stan) often use more advanced MCMC variants, such as **Hamiltonian Monte Carlo (HMC)** and the **No-U-Turn Sampler (NUTS)**. HMC uses gradient information to simulate physical dynamics (like a puck sliding over a frictionless surface shaped by the posterior density) to propose new states far away from the current state, drastically improving efficiency in high dimensions [3]. However, HMC still relies heavily on randomness: it randomly samples initial "momentum" variables at each step and uses floating-point intensive numerical integration (Leapfrog steps) to simulate the physical trajectory [3].

## Why MCMC is Highly Sensitive to Environmental Differences

Because MCMC is fundamentally a sequential, random walk through a high-dimensional space, it is exceptionally vulnerable to the non-reproducibility factors we discussed regarding Docker images and hardware.

### 1. The Butterfly Effect of Floating-Point Math
In MCMC, every step depends strictly on the previous step. If a floating-point non-associativity issue (caused by a different BLAS library, a different `glibc` version, or a different GPU architecture) alters the calculation of the log-likelihood or the gradient by even a single bit, the acceptance ratio will change slightly. 

This microscopic difference might cause a proposed state to be rejected on Machine A but accepted on Machine B. Once the chains diverge at a single step, their subsequent random walks will follow entirely different paths through the parameter space [4]. 

### 2. NUTS and Leapfrog Integration
The NUTS algorithm, the default in PyMC and Stan, uses a recursive algorithm to build a trajectory of "leapfrog" integration steps until the path makes a "U-turn" [3]. The exact point at which a U-turn is detected depends on the accumulation of floating-point calculations. A tiny numerical difference across different Docker environments can cause the NUTS algorithm to take one more or one fewer leapfrog step, completely changing the proposed parameter state for that iteration [4].

### 3. Parallel Chains and Threading
To ensure MCMC has converged to the true posterior, practitioners run multiple independent Markov chains in parallel and compare them (using diagnostics like the Gelman-Rubin $\hat{R}$ statistic) [5]. If the underlying linear algebra libraries (like OpenBLAS or Intel MKL) use different multi-threading strategies across different environments, the order of operations in matrix multiplications will change, breaking determinism even if the global random seed is fixed [4].

## Reproducibility vs. Statistical Validity

When a Bayesian model yields different MCMC traces on a local machine versus a CI/CD Docker container, it is a source of immense frustration. However, it is crucial to distinguish between **bitwise reproducibility** and **statistical validity**.

If an MCMC algorithm has truly converged, the chains from Machine A and Machine B will look different step-by-step, but they are sampling from the *exact same underlying posterior distribution*. 
*   The mean estimates of the parameters should be nearly identical.
*   The credible intervals (uncertainty bounds) should match.
*   The posterior predictive checks should yield the same conclusions.

If running the same code on two different machines yields statistically *different* posterior distributions (e.g., different means or non-overlapping credible intervals), it usually indicates that the MCMC chains have not converged, the model is poorly specified, or the burn-in period was insufficient—not just a floating-point discrepancy [5].

## Alternatives: Variational Inference (VI)

Because MCMC is computationally expensive and difficult to scale to massive datasets (like those used in Bayesian Deep Learning), practitioners sometimes turn to **Variational Inference (VI)**. 

Instead of using randomness to sample from the intractable posterior, VI frames Bayesian inference as an optimization problem. It proposes a simple, parameterized distribution (like a Gaussian) and uses gradient descent to adjust its parameters until it closely matches the true posterior (minimizing the Kullback-Leibler divergence). 

While VI is faster and often more deterministic (as it relies on standard gradient descent rather than sequential random walks), it is an approximation. VI tends to underestimate posterior variance (uncertainty) and cannot capture complex, multi-modal posterior shapes as accurately as MCMC. Therefore, MCMC remains the gold standard for rigorous Bayesian inference when computational resources allow.

---

### References

[1] Columbia University. "MCMC and Bayesian Modeling." http://www.columbia.edu/~mh2078/MachineLearningORFE/MCMC_Bayes.pdf

[2] Tweag. "Markov chain Monte Carlo (MCMC) Sampling, Part 1: The Basics." https://www.tweag.io/blog/2019-10-25-mcmc-intro1/

[3] Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo." Journal of Machine Learning Research.

[4] Stan Discourse. "Same code (with the same seed) but different results on different platforms? Why?" https://discourse.mc-stan.org/t/same-code-with-the-same-seed-but-different-results-on-different-platforms-why/24141

[5] Vehtari, A., et al. (2020). "Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC." arXiv preprint.
