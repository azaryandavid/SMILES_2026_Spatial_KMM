import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

def generate_spatial_data(n_samples=300, func='sin_cos', noise_std=0.3, sigma=1.0, length_scale=1.0, coord_range=(-5, 5)):
    X = np.random.uniform(coord_range[0], coord_range[1], size=(n_samples, 2))
    if func == 'linear':
        y = (X[:, 0] + X[:, 1] * 0.5 +
             np.random.normal(0, noise_std, n_samples))
    elif func == 'exponential':
        y = (np.exp(X[:, 0] * 0.3) + np.sin(X[:, 1] * 2) +
             np.random.normal(0, noise_std, n_samples))
    elif func == 'periodic':
        y = (np.sin(X[:, 0] * 2) * np.cos(X[:, 1] * 2) +
             np.random.normal(0, noise_std, n_samples))
    elif func == 'random':
        y = np.random.uniform(-5, 5, size=n_samples)
    elif func == 'random_autocorr':
        K = sigma**2 * rbf_kernel(X, X, gamma=1/(2*length_scale**2))
        K += 1e-6 * np.eye(n_samples)
        mean = np.zeros(n_samples)
        rng = np.random.RandomState(42)
        y = rng.multivariate_normal(mean, K)
    else:
        raise ValueError("Undefined function")
    return X, y

def random_cv(X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    errors = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        errors.append(mean_squared_error(y_te, y_pred))
    return np.mean(errors), np.std(errors)

def spatial_cv(X, y, n_folds=5):
    kmeans = KMeans(n_clusters=n_folds, random_state=42, n_init=10)
    fold_ids = kmeans.fit_predict(X)
    errors = []
    for test_fold in range(n_folds):
        test_idx = np.where(fold_ids == test_fold)[0]
        train_idx = np.where(fold_ids != test_fold)[0]
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        errors.append(mean_squared_error(y_te, y_pred))
    return np.mean(errors), np.std(errors)

def kde_importance_weights(X_train, X_test, clip_limit=10.0):
    try:
        kde_train = gaussian_kde(X_train.T)
        kde_test = gaussian_kde(X_test.T)
        p_train = kde_train.evaluate(X_train.T)
        p_test = kde_test.evaluate(X_train.T)
        weights = p_test / p_train
        weights = np.nan_to_num(weights, nan=1.0, posinf=clip_limit, neginf=1.0)
        weights = np.clip(weights, 0, clip_limit)
        return weights
    except np.linalg.LinAlgError:
        print("KDE is failed.")
        return np.ones(len(X_train))

def importance_weighted_spatial_cv(X, y, n_folds=5, clip_limit=10.0):
    kmeans = KMeans(n_clusters=n_folds, random_state=42, n_init=10)
    fold_ids = kmeans.fit_predict(X)
    errors = []
    for test_fold in range(n_folds):
        test_idx = np.where(fold_ids == test_fold)[0]
        train_idx = np.where(fold_ids != test_fold)[0]
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        weights = kde_importance_weights(X_tr, X_te, clip_limit=clip_limit)
        weights = weights / np.sum(weights) * len(X_tr)
        model = LinearRegression()
        model.fit(X_tr, y_tr, sample_weight=weights)
        y_pred = model.predict(X_te)
        errors.append(mean_squared_error(y_te, y_pred))
    return np.mean(errors), np.std(errors)

def kmm_weights(X_train, X_test, gamma=1.0, B=10):
    n_train = X_train.shape[0]
    K = rbf_kernel(X_train, X_train, gamma=gamma)
    kappa = np.mean(rbf_kernel(X_train, X_test, gamma=gamma), axis=1)

    def objective(w):
        return 0.5 * w @ K @ w - kappa @ w

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - n_train}]
    bounds = [(0, B)] * n_train
    initial = np.ones(n_train)

    res = minimize(objective, initial, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 1000, 'ftol': 1e-8})
    if not res.success:
        print("KMM didn't converge.")
        return np.ones(n_train)
    return res.x

def sample_uniform_target(coord_range=(-5, 5), n_target=1000, random_state=0):
    rng = np.random.RandomState(random_state)
    return rng.uniform(coord_range[0], coord_range[1], size=(n_target, 2))


def _resolve_weights(weighting, X_tr, X_te, train_idx, global_weights, gamma, B):
    if weighting == 'fold':
        return kmm_weights(X_tr, X_te, gamma=gamma, B=B)
    elif weighting == 'global':
        return global_weights[train_idx]
    else:
        return "oops"


def kmm_weighted_spatial_cv(X, y, n_folds=5, gamma=0.5, B=10, weighting='fold', X_target=None):
    kmeans = KMeans(n_clusters=n_folds, random_state=42, n_init=10)
    fold_ids = kmeans.fit_predict(X)

    global_weights = None
    if weighting == 'global':
        if X_target is None:
            X_target = sample_uniform_target()
        global_weights = kmm_weights(X, X_target, gamma=gamma, B=B)

    errors = []
    for test_fold in range(n_folds):
        test_idx = np.where(fold_ids == test_fold)[0]
        train_idx = np.where(fold_ids != test_fold)[0]
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        weights = _resolve_weights(weighting, X_tr, X_te, train_idx, global_weights, gamma, B)
        model = LinearRegression()
        model.fit(X_tr, y_tr, sample_weight=weights)
        y_pred = model.predict(X_te)
        errors.append(mean_squared_error(y_te, y_pred))
    return np.mean(errors), np.std(errors)


def kmm_weighted_random_cv(X, y, n_folds=5, gamma=0.5, B=10, weighting='fold', X_target=None):
    # G2 extension
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    global_weights = None
    if weighting == 'global':
        if X_target is None:
            X_target = sample_uniform_target()
        global_weights = kmm_weights(X, X_target, gamma=gamma, B=B)

    errors = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        weights = _resolve_weights(weighting, X_tr, X_te, train_idx, global_weights, gamma, B)
        model = LinearRegression()
        model.fit(X_tr, y_tr, sample_weight=weights)
        y_pred = model.predict(X_te)
        errors.append(mean_squared_error(y_te, y_pred))
    return np.mean(errors), np.std(errors)

def run_comparison(n_folds=5, n_samples=300, noise_std=0.3, gamma=0.5, B=10):
    funcs = ['linear', 'exponential', 'periodic', 'random', 'random_autocorr']
    results = []
    for func in funcs:
        X, y = generate_spatial_data(n_samples=n_samples, func=func,
                                     noise_std=noise_std, coord_range=(-5, 5))
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        plt.figure(figsize=(6, 5))
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_scaled, cmap='viridis')
        plt.colorbar(label='y')
        plt.title(f'Normalized {func}')
        plt.show()

        mse_rand, std_rand = random_cv(X_scaled, y_scaled, n_folds=n_folds)
        mse_spatial, std_spatial = spatial_cv(X_scaled, y_scaled, n_folds=n_folds)
        mse_imp, std_imp = importance_weighted_spatial_cv(X_scaled, y_scaled, n_folds=n_folds, clip_limit=10.0)
        mse_kmm, std_kmm = kmm_weighted_spatial_cv(X_scaled, y_scaled, n_folds=n_folds, gamma=gamma, B=B)

        results.append({
            'Function': func,
            'Random CV_mean': mse_rand,
            'Random CV_std': std_rand,
            'Spatial CV_mean': mse_spatial,
            'Spatial CV_std': std_spatial,
            'Importance-weighted CV_mean': mse_imp,
            'Importance-weighted CV_std': std_imp,
            'KMM-weighted spatial CV_mean': mse_kmm,
            'KMM-weighted spatial CV_std': std_kmm
        })

    df = pd.DataFrame(results)
    display_df = df.copy()
    display_df['Random CV'] = display_df['Random CV_mean'].map('{:.4f}'.format) + ' ± ' + display_df['Random CV_std'].map('{:.4f}'.format)
    display_df['Spatial CV'] = display_df['Spatial CV_mean'].map('{:.4f}'.format) + ' ± ' + display_df['Spatial CV_std'].map('{:.4f}'.format)
    display_df['Importance-weighted CV'] = display_df['Importance-weighted CV_mean'].map('{:.4f}'.format) + ' +- ' + display_df['Importance-weighted CV_std'].map('{:.4f}'.format)
    display_df['KMM-weighted spatial CV'] = display_df['KMM-weighted spatial CV_mean'].map('{:.4f}'.format) + ' +- ' + display_df['KMM-weighted spatial CV_std'].map('{:.4f}'.format)
    display_df = display_df[['Function', 'Random CV', 'Spatial CV', 'Importance-weighted CV', 'KMM-weighted spatial CV']]

    return results, display_df
