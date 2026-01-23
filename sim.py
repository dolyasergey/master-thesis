#simulations

# imports
import pandas as pd
import numpy as np

# Simulate brownian motions
def sim_dW(n_steps, dt, corr, n_paths = 1000, seed = 99):
    n_factors = corr.shape[0]
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_paths, n_steps, n_factors))
    cholesky_matrix = np.linalg.cholesky(corr)
    Z_corr = Z @ cholesky_matrix.T
    dW = np.sqrt(dt) * Z_corr
    return dW

### DETERMENISTIC ###

#Sigma
def sim_sigma(n_steps, dt, sigma_0, kappa_1, sigma_hat, n_paths=1000):
    sigma_flat = np.empty(n_steps, dtype=float)
    sigma_flat[0] = sigma_0
    for idx in range(1, n_steps):
        sigma_m = sigma_flat[idx - 1]
        sigma_flat[idx] = sigma_m + kappa_1 * (sigma_hat - sigma_m) * dt
    return np.tile(sigma_flat.reshape(1, n_steps), (n_paths, 1))

#Eta
def sim_eta(n_steps, dt, eta_0, kappa_2, eta_hat, n_paths = 1000):
    eta_flat = np.empty(n_steps, dtype=float)
    eta_flat[0] = eta_0

    for idx in range(1, n_steps):
        eta_m = eta_flat[idx - 1]
        eta_flat[idx] = eta_m + kappa_2 * (eta_hat - eta_m) * dt

    etas = np.tile(eta_flat.reshape(1, n_steps), (n_paths, 1))
    return etas

def sim_mu(n_steps, dt, mu_0, mu_hat, kappa_mu, etas, dW_mu):

    n_paths = dW_mu.shape[0]

    mu = np.empty((n_paths, n_steps), dtype=float)
    mu[:, 0] = mu_0

    for idx in range(1, n_steps):
        mu_m = mu[:, idx - 1]
        mu[:, idx] = mu_m + kappa_mu * (mu_hat - mu_m) * dt + etas[:, idx - 1] * dW_mu[:, idx - 1]

    return mu

def sim_revenue(n_steps, dt, R_0, mus, sigmas, dW_R):

    n_paths = dW_R.shape[0]

    R = np.empty((n_paths, n_steps), dtype=float)
    R[:, 0] = R_0

    for idx in range(1, n_steps):
        R_m = R[:, idx - 1]
        R[:, idx] = R_m * (1.0 + mus[:, idx - 1] * dt + sigmas[:, idx - 1] * dW_R[:, idx - 1])

    return R

