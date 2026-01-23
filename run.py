#imports 
import numpy as np
import pandas as pd

from helper import rho_matrix_construct
from sim import sim_dW, sim_sigma, sim_eta, sim_mu, sim_revenue

def run_sim(params, n_steps, n_paths, T, seed=99, return_all = False):
    # --- read scalar params
    dt = T / n_steps

    # --- correlation matrix (3 factors by convention)
    if "rhos" in params:
        rhos = np.array(params["rhos"], dtype=float)
    else:
        rhos = np.array([float(params["rho12"]), float(params["rho13"]), float(params["rho23"])], dtype=float)

    corr = rho_matrix_construct(3, rhos)

    # --- correlated Brownian increments
    dW = sim_dW(n_paths=n_paths, n_steps=n_steps, dt=dt, corr=corr, seed=seed)
    dW_R = dW[:, :, 0]
    dW_mu = dW[:, :, 1]
    # dW_r = dW[:, :, 2]  # rfor interest rates

    # --- deterministic sigma and eta (tiled to paths)
    sigmas = sim_sigma(
        n_steps=n_steps,
        dt=dt,
        sigma_0=float(params["sigma_0"]),
        kappa_1=float(params["kappa_1"]),
        sigma_hat=float(params["sigma_hat"]),
        n_paths=n_paths,
    )

    etas = sim_eta(
        n_steps=n_steps,
        dt=dt,
        eta_0=float(params["eta_0"]),
        kappa_2=float(params["kappa_2"]),
        eta_hat=float(params.get("eta_hat", 0.0)),  # default zero-reverting
        n_paths=n_paths,
    )

    # --- mu and revenue paths
    mus = sim_mu(
        n_steps=n_steps,
        dt=dt,
        mu_0=float(params["mu_0"]),
        mu_hat=float(params["mu_hat"]),
        kappa_mu=float(params["kappa_mu"]),
        etas=etas,
        dW_mu=dW_mu,
    )

    R = sim_revenue(
        n_steps=n_steps,
        dt=dt,
        R_0=float(params["R_0"]),
        mus=mus,
        sigmas=sigmas,
        dW_R=dW_R,
    )

    if return_all:
        return {
            "R": R,
            "mus": mus,
            "sigmas": sigmas,
            "etas": etas,
            "dW": dW,
            "dt": dt,
        }

    return R

def sim_PL(R, F, theta):
    C = F + theta * R
    Y = R - C
    return C, Y
