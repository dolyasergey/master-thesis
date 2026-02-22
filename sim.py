import pandas as pd
import numpy as np

def sim_dW(n_steps, dt, corr, n_paths=1000, seed=99):
    n_factors = corr.shape[0]
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_paths, n_steps, n_factors))
    cholesky_matrix = np.linalg.cholesky(corr)
    Z_corr = Z @ cholesky_matrix.T
    dW = np.sqrt(dt) * Z_corr
    return dW

def sim_mu(n_steps, dt, mu_0, mu_hat, kappa_mu, eta, dW_mu):
    n_paths = dW_mu.shape[0]
    mu = np.empty((n_paths, n_steps), dtype=float)
    mu[:, 0] = mu_0
    for idx in range(1, n_steps):
        mu_m = mu[:, idx - 1]
        mu[:, idx] = mu_m + kappa_mu * (mu_hat - mu_m) * dt + eta * dW_mu[:, idx - 1]
    return mu

def sim_revenue(n_steps, dt, R_0, mus, sigma, dW_R):
    n_paths = dW_R.shape[0]
    R = np.empty((n_paths, n_steps), dtype=float)
    R[:, 0] = R_0
    for idx in range(1, n_steps):
        R_m = R[:, idx - 1]
        R[:, idx] = R_m * (1.0 + mus[:, idx - 1] * dt + sigma * dW_R[:, idx - 1])
    return R

def sim_PL(R, F, theta, gamma, tau):
    C = F + (theta + gamma) * R
    Y = (R - C) * (1.0 - tau)
    return C, Y

def sim_firm_value(Y, dt, rho, mu_tv):
    if rho <= mu_tv:
        raise ValueError(f"Gordon-Shapiro violated: rho={rho:.4f} must exceed mu_tv={mu_tv:.4f}.")

    n_paths, n_steps = Y.shape
    T_idx = n_steps - 1

    V = np.zeros((n_paths, n_steps), dtype=float)
    V[:, T_idx] = Y[:, T_idx] * (1.0 + mu_tv) / (rho - mu_tv)

    disc = np.exp(-rho * dt)
    for t in range(T_idx - 1, -1, -1):
        V[:, t] = disc * (Y[:, t + 1] * dt + V[:, t + 1])

    return V

def sim_cir(n_steps, dt, r0, xi, theta_r, omega, dW_r):
    n_paths = dW_r.shape[0]
    r = np.empty((n_paths, n_steps), dtype=float)
    r[:, 0] = r0
    for t in range(1, n_steps):
        r_prev = r[:, t - 1]
        r_pos = np.maximum(r_prev, 0.0)
        r[:, t] = r_prev + xi * (theta_r - r_pos) * dt + omega * np.sqrt(r_pos) * dW_r[:, t - 1]
    return np.maximum(r, 0.0)

def sim_LBO(Y, V_entry, V_exit, alpha, r, return_equity_cf=True):
    n_paths, n_steps = Y.shape

    if np.isscalar(r):
        r_mat = np.full((n_paths, n_steps), float(r))
    else:
        r_mat = np.asarray(r, dtype=float)
        if r_mat.shape == (n_paths, n_steps - 1):
            r_mat = np.concatenate([r_mat, r_mat[:, -1][:, None]], axis=1)

    D = np.empty((n_paths, n_steps), dtype=float)
    I = np.zeros((n_paths, n_steps), dtype=float)
    P = np.zeros((n_paths, n_steps), dtype=float)
    defaulted = np.zeros((n_paths, n_steps), dtype=bool)
    equity_cf = np.zeros((n_paths, n_steps), dtype=float) if return_equity_cf else None

    D[:, 0] = alpha * V_entry
    alive = np.ones(n_paths, dtype=bool)

    for t in range(n_steps):
        D_prev = D[:, t]
        interest_due = r_mat[:, t] * D_prev
        cash = Y[:, t]
        will_default = alive & (cash < interest_due)
        if np.any(will_default):
            defaulted[will_default, t:] = True
            alive[will_default] = False
        alive_idx = alive
        I[alive_idx, t] = interest_due[alive_idx]
        rem = cash[alive_idx] - interest_due[alive_idx]
        P_pay = np.minimum(D_prev[alive_idx], rem)
        P[alive_idx, t] = P_pay
        if return_equity_cf:
            equity_cf[alive_idx, t] = rem - P_pay
        if t < n_steps - 1:
            D[:, t + 1] = D_prev
            D[alive_idx, t + 1] = D_prev[alive_idx] - P_pay

    if return_equity_cf:
        T_idx = n_steps - 1
        equity_cf[alive, T_idx] += np.maximum(V_exit[alive] - D[alive, T_idx], 0.0)

    return D, I, P, defaulted, equity_cf