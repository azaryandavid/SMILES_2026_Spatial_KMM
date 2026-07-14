# SMILES-2026 Spatial_KMM
Used environment:
Python 3.12.6
numpy 2.4.2
json 2.0.9
matplotlib 3.10.0
scikit-learn 1.6.1
scipy 1.17.1
tabulate 0.10.0
re 2.2.1

The main solution is in the file `KMM.ipynb`. It used `Jupyter Notebook` to run the `KMM.ipynb`, the main functions for tasks are in the file `CV_Methods.py`, 
the methods realization are in the `Metthods.py`. 
To run the project: Run all cells of the `KMM.ipynb`.
Runtime: ~1 min.

## Tasks of the project
1. Formalize the estimand for KMM-weighted spatial cross-validation.
2. Implement global and fold-specific KMM weighting for validation losses.
3. Compare random CV, spatial CV, importance-weighted CV, and KMM-weighted spatial CV on controlled spatial experiments.
4. Report diagnostics for effective sample size, weight concentration, support overlap, and sensitivity to kernel bandwidth and block design.
   
## Methods
`Random CV`
Random folds with simple Linear Regression Model

`Spatial CV`
KMeans clusterisation with Linear Regression Model

`Importance-weighted CV`
KMeans clusterisation with weights calculated by KDE gaussian method in `kde_importance_weights` function 

`KMM-weighted spatial CV`
KMeans clusterization with weights calculated by optimization taks with function $\frac{1}{2}w^{T}Kw-\kappa^{T}w$
realized in `kmm_weights`

## Metrics
Realised in code: simple MSE. But more effective metrics are:
* *Risk-estimation bias*: difference between estimated CV error and known deployment error in simulations.
* *Absolute and relative bias*: $|{R̂}_{CV} - R_{dep}|$ and the same error normalized by $R_{dep}$.
* *Variance and stability*: variability across repeated fold partitions and KMM hyperparameters.
* *Overlap quality*: effective sample size, maximum weight, clipped-weight share, and MMD before and after weighting.
* *Model-selection impact*: whether weighted spatial CV selects hyperparameters that improve deployment error.
* *Failure detection*: whether low effective sample size, high clipping rate, or residual MMD predict cases where the weighted estimate is unreliable.

## Results
Results are imported from `results.json`.

<!-- TABLE_START -->
| Function        | Random CV       | Spatial CV      | Importance-weighted CV   | KMM-weighted spatial CV   |
|:----------------|:----------------|:----------------|:-------------------------|:--------------------------|
| linear          | 0.0083 ± 0.0012 | 0.0084 ± 0.0013 | 0.0100 ± 0.0012          | 0.0095 ± 0.0018           |
| exponential     | 0.3345 ± 0.0294 | 0.4493 ± 0.1568 | 0.4242 ± 0.0772          | 0.4654 ± 0.1376           |
| periodic        | 1.0159 ± 0.2376 | 1.0082 ± 0.1484 | 1.3351 ± 0.3398          | 1.0403 ± 0.1579           |
| random          | 1.0276 ± 0.2354 | 1.0128 ± 0.1178 | 1.5315 ± 0.7885          | 1.0756 ± 0.1715           |
| random_autocorr | 1.0067 ± 0.1173 | 1.9548 ± 1.0955 | 1.6816 ± 0.6645          | 1.6155 ± 1.4618           |
<!-- TABLE_END -->

## G2 results

`KMM-spatial (fold)` re-solves the KMM program per test fold, matching that fold's train covariates to its own held-out covariates. `KMM-spatial (global)` solves KMM once for the whole sample against a fixed proxy of the deployment distribution (uniform coverage of the domain) and reuses those weights across folds. The `-random` variants apply the same two weighting schemes on top of random (non-spatial) folds, as a control.

| Function        | Random CV        | Spatial CV       | KMM-spatial (fold)   | KMM-spatial (global)   | KMM-random (fold)   | KMM-random (global)   |
|:----------------|:-----------------|:-----------------|:---------------------|:-----------------------|:--------------------|:----------------------|
| linear          | 0.0106 +- 0.0009 | 0.0119 +- 0.0029 | 0.0124 +- 0.0007     | 0.0112 +- 0.0021       | 0.0119 +- 0.0012    | 0.0120 +- 0.0013      |
| exponential     | 0.3645 +- 0.0502 | 0.3946 +- 0.0399 | 0.4230 +- 0.0544     | 0.4252 +- 0.0651       | 0.4590 +- 0.0148    | 0.4512 +- 0.0128      |
| periodic        | 1.0233 +- 0.2093 | 1.0289 +- 0.1059 | 1.2362 +- 0.2611     | 1.2322 +- 0.2933       | 1.0733 +- 0.2339    | 1.0999 +- 0.2158      |
| random          | 1.0043 +- 0.0549 | 1.0387 +- 0.1443 | 1.2510 +- 0.1861     | 1.3382 +- 0.2845       | 1.0798 +- 0.1126    | 1.0728 +- 0.0925      |
| random_autocorr | 0.9848 +- 0.1661 | 1.0730 +- 0.6818 | 1.4286 +- 1.0597     | 1.7685 +- 1.0497       | 1.1705 +- 0.1676    | 1.1803 +- 0.1695      |
