# SMILES-2026 Spatial_KMM
Used environment:
Python 3.12.6
numpy 2.4.2
json 2.0.9
matplotlib 3.10.0
scikit-learn 1.6.1
scipy 1.17.1

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
* *Absolute and relative bias*: $|\hat{R}_{CV} - R_{dep}|$ and the same error normalized by $R_{dep}$.
* *Variance and stability*: variability across repeated fold partitions and KMM hyperparameters.
* *Overlap quality*: effective sample size, maximum weight, clipped-weight share, and MMD before and after weighting.
* *Model-selection impact*: whether weighted spatial CV selects hyperparameters that improve deployment error.
* *Failure detection*: whether low effective sample size, high clipping rate, or residual MMD predict cases where the weighted estimate is unreliable.

## Results
Results are imported from `results.json`.

| Function          | Random CV         | Spatial CV        | Importance-weighted CV | KMM-weighted spatial CV |
|-------------------|-------------------|-------------------|------------------------|--------------------------|
| linear            | 0.0067 ± 0.0009   | 0.0067 ± 0.0009   | 0.0078 ± 0.0014        | 0.0070 ± 0.0011          |
| exponential       | 0.3837 ± 0.0299   | 0.4833 ± 0.1180   | 0.7498 ± 0.5784        | 0.4931 ± 0.1595          |
| periodic          | 1.0263 ± 0.1024   | 1.0235 ± 0.2113   | 1.1595 ± 0.2280        | 1.1656 ± 0.2265          |
| random            | 1.0052 ± 0.1578   | 1.0102 ± 0.0314   | 1.3053 ± 0.3765        | 1.0697 ± 0.0849          |
| random_autocorr   | 0.9454 ± 0.1790   | 1.2316 ± 0.5302   | 2.5865 ± 1.7302        | 1.1797 ± 0.2493          |