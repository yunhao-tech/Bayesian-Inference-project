# Bayesian Inference project

Based on the paper [Hierarchical Bayesian Analysis of the Seemingly Unrelated Regression and Simultaneous
Equations Models Using a Combination of Direct Monte Carlo and Importance Sampling Techniques](https://projecteuclid.org/journals/bayesian-analysis/volume-5/issue-1/Hierarchical-Bayesian-analysis-of-the-seemingly-unrelated-regression-and-simultaneousequations/10.1214/10-BA503.full), we reproduce some of its results.

The paper considers the SUR (seemingly unrelated regression) problem and its transformed version. It proposes a DMC-IS method (direct monte carlo with importance sampling) to simulate the parameters in SUR.

Here, we implement and compare three methods:

- MCMC on original SUR
- DMC on transformed SUR
- DMC-IS on transformed SUR


