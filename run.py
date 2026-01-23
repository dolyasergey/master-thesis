#imports 
import numpy as np
import pandas as pd

from helper import rho_matrix_construct
from sim import *

def run_sim(params, n_steps, n_paths, T, alpha, entry_mult, seed=99, stochastic_ir = False, return_all = False):
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
    dW_r = dW[:, :, 2]  # rfor interest rates

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

    #costs

    C, Y = sim_PL(
        R=R,
        F=float(params["F_0"]),
        theta=float(params["theta"])
    )

    # interest rates

    if stochastic_ir:
        r = sim_cir(
            n_steps=n_steps, 
            dt=dt, 
            r0=params['r0'], 
            xi=params['xi'], 
            theta_r=params['theta_r'], 
            omega=params['omega'], 
            dW_r=dW_r)
    else:
        r = 0.06


    #LBO

    EBITDA_0 = params['R_0'] * (1 - params['theta']) - params['F_0']
    D, I, P, defaulted, eq_cf = sim_LBO(
    Y=Y,
    V0=EBITDA_0 * entry_mult,
    alpha=alpha,
    r=r,
    return_equity_cf=True
)

    if return_all:
        return {
            "R": R,
            'C': C,
            'Y': Y,
            "mus": mus,
            "sigmas": sigmas,
            "etas": etas,
            "dW": dW,
            "dt": dt,
            'D': D, 
            'I': I,
            'P': P,
            'defaulted': defaulted,
            'CF': eq_cf,
            'r': r
        }

    return eq_cf



