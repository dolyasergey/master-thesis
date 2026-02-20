import numpy as np
import pandas as pd

from helper import rho_matrix_construct
from sim import sim_dW, sim_mu, sim_revenue, sim_PL, sim_firm_value, sim_cir, sim_LBO

def run_sim(params, n_steps, n_paths, T, alpha, seed=99, stochastic_ir=False, return_all=False):
    dt = T / n_steps

    rho12    = float(params.get("rho12", 0.0))
    corr     = rho_matrix_construct(3, np.array([rho12, 0.0, 0.0]))
    dW       = sim_dW(n_paths=n_paths, n_steps=n_steps, dt=dt, corr=corr, seed=seed)
    dW_R     = dW[:, :, 0]
    dW_mu    = dW[:, :, 1]
    dW_r     = dW[:, :, 2]

    sigma    = float(params["sigma"])
    eta      = float(params["eta"])
    tau      = float(params["tau"])
    rho      = float(params["rho"])
    mu_hat   = float(params["mu_hat"])    # mean-reversion target for growth SDE
    mu_tv    = float(params["mu_tv"])     # terminal value perpetuity growth rate

    mus = sim_mu(
        n_steps=n_steps, dt=dt,
        mu_0=float(params["mu_0"]),
        mu_hat=mu_hat,
        kappa_mu=float(params["kappa_mu"]),
        eta=eta,
        dW_mu=dW_mu,
    )

    R = sim_revenue(
        n_steps=n_steps, dt=dt,
        R_0=float(params["R_0"]),
        mus=mus,
        sigma=sigma,
        dW_R=dW_R,
    )

    C, Y = sim_PL(R=R, F=float(params["F_0"]), theta=float(params["theta"]), tau=tau)

    if stochastic_ir:
        r = sim_cir(
            n_steps=n_steps, dt=dt,
            r0=float(params["r0"]),
            xi=float(params["xi"]),
            theta_r=float(params["theta_r"]),
            omega=float(params["omega"]),
            dW_r=dW_r,
        )
    else:
        r = float(params.get("r_bar", 0.06))

    # Firm value under model: entry at t=0, exit terminal value at t=T
    V = sim_firm_value(Y=Y, dt=dt, rho=rho, mu_tv=mu_tv)
    V_entry = V[:, 0]         # model-implied value at entry, per path
    V_exit  = V[:, -1]        # Gordon-Shapiro terminal value at exit, per path

    D, I, P, defaulted, eq_cf = sim_LBO(
        Y=Y,
        V_entry=V_entry,
        V_exit=V_exit,
        alpha=alpha,
        r=r,
        return_equity_cf=True,
    )

    if return_all:
        return {
            "R": R, "C": C, "Y": Y, "mus": mus,
            "sigma": sigma, "eta": eta, "dW": dW, "dt": dt,
            "V": V, "V_entry": V_entry, "V_exit": V_exit,
            "D": D, "I": I, "P": P, "defaulted": defaulted, "CF": eq_cf, "r": r,
        }

    return eq_cf