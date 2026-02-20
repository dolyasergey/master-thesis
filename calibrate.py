import numpy as np

from sim import sim_dW, sim_mu, sim_revenue, sim_PL, sim_firm_value


def _run_and_get_V0(params, sigma, eta, n_steps, n_paths, dt, rho, mu_tv, tau, dW,
                    R0_override=None, mu0_override=None):
    p = dict(params)
    if R0_override  is not None: p["R_0"]  = R0_override
    if mu0_override is not None: p["mu_0"] = mu0_override

    mu = sim_mu(n_steps, dt, float(p["mu_0"]), float(p["mu_hat"]),
                float(p["kappa_mu"]), eta, dW[:, :, 1])
    R  = sim_revenue(n_steps, dt, float(p["R_0"]), mu, sigma, dW[:, :, 0])
    theta_total = float(params["theta"]) + float(params.get("gamma", 0.0))
    _, Y = sim_PL(R=R, F=float(p["F_0"]), theta=theta_total, tau=tau)
    V = sim_firm_value(Y=Y, dt=dt, rho=rho, mu_tv=mu_tv)
    return V[:, 0].mean()


def model_equity_vol(eta_candidate, sigma, params, n_steps, n_paths, T,
                     rho, mu_tv, tau, dv_ratio=0.1773, dW=None, seed=99,
                     bump_R_frac=0.01, bump_mu=0.001):

    if not 0.0 <= dv_ratio < 1.0:
        raise ValueError(f"dv_ratio must be in [0, 1), got {dv_ratio:.4f}.")

    dt = T / n_steps

    if dW is None:
        rho12 = float(params.get("rho12", 0.0))
        corr  = np.array([[1.0, rho12], [rho12, 1.0]])
        dW = sim_dW(n_paths=n_paths, n_steps=n_steps, dt=dt, corr=corr, seed=seed)

    R0  = float(params["R_0"])
    mu0 = float(params["mu_0"])

    V_base   = _run_and_get_V0(params, sigma, eta_candidate, n_steps, n_paths,
                                dt, rho, mu_tv, tau, dW)
    V_R_up   = _run_and_get_V0(params, sigma, eta_candidate, n_steps, n_paths,
                                dt, rho, mu_tv, tau, dW,
                                R0_override=R0 * (1.0 + bump_R_frac))
    V_R_dn   = _run_and_get_V0(params, sigma, eta_candidate, n_steps, n_paths,
                                dt, rho, mu_tv, tau, dW,
                                R0_override=R0 * (1.0 - bump_R_frac))
    V_mu_up  = _run_and_get_V0(params, sigma, eta_candidate, n_steps, n_paths,
                                dt, rho, mu_tv, tau, dW,
                                mu0_override=mu0 + bump_mu)
    V_mu_dn  = _run_and_get_V0(params, sigma, eta_candidate, n_steps, n_paths,
                                dt, rho, mu_tv, tau, dW,
                                mu0_override=mu0 - bump_mu)

    dVdR  = (V_R_up  - V_R_dn)  / (2.0 * R0  * bump_R_frac)
    dVdmu = (V_mu_up - V_mu_dn) / (2.0 * bump_mu)

    eps_R  = dVdR  * R0  / V_base
    eps_mu = dVdmu       / V_base

    sigma_V2 = (eps_R * sigma)**2 + (eps_mu * eta_candidate)**2
    sigma_V  = float(np.sqrt(sigma_V2))
    sigma_E  = sigma_V / (1.0 - dv_ratio)

    return sigma_E


def calibrate_eta(sigma_market, sigma, params, n_steps, n_paths, T,
                  rho, mu_tv, tau,
                  dv_ratio=0.1773,
                  eta_grid=None,
                  seed=99, verbose=True):

    if eta_grid is None:
        eta_grid = np.arange(0.005, 0.505, 0.005)
    eta_grid = np.asarray(eta_grid, dtype=float)

    dt = T / n_steps
    rho12 = float(params.get("rho12", 0.0))
    corr  = np.array([[1.0, rho12], [rho12, 1.0]])
    dW = sim_dW(n_paths=n_paths, n_steps=n_steps, dt=dt, corr=corr, seed=seed)

    n_points     = len(eta_grid)
    sigma_E_grid = np.empty(n_points, dtype=float)

    if verbose:
        print(f"Grid search: {n_points} points, "
              f"eta in [{eta_grid[0]:.3f}, {eta_grid[-1]:.3f}]")
        print(f"Target sigma_market = {sigma_market:.4f}  |  D/V = {dv_ratio:.4f}\n")
        print(f"  {'#':>4}  {'eta':>7}  {'sigma_E':>9}  {'|error|':>9}")
        print(f"  {'-'*4}  {'-'*7}  {'-'*9}  {'-'*9}")

    for i, eta in enumerate(eta_grid):
        se = model_equity_vol(
            eta_candidate=float(eta),
            sigma=sigma,
            params=params,
            n_steps=n_steps, n_paths=n_paths, T=T,
            rho=rho, mu_tv=mu_tv, tau=tau,
            dv_ratio=dv_ratio, dW=dW, seed=seed,
        )
        sigma_E_grid[i] = se
        if verbose:
            print(f"  {i+1:>4}  {eta:>7.4f}  {se:>9.4f}  {abs(se - sigma_market):>9.4f}")

    abs_errors = np.abs(sigma_E_grid - sigma_market)
    best_idx   = int(np.argmin(abs_errors))
    eta_star   = float(eta_grid[best_idx])
    sigma_star = float(sigma_E_grid[best_idx])

    result_info = {
        "eta_star":            eta_star,
        "sigma_E_at_eta_star": sigma_star,
        "sigma_market":        sigma_market,
        "dv_ratio":            dv_ratio,
        "eta_grid":            eta_grid,
        "sigma_E_grid":        sigma_E_grid,
        "abs_errors":          abs_errors,
    }

    if verbose:
        print(f"\n{'='*55}")
        print(f"  eta*                     = {eta_star:.4f}")
        print(f"  sigma_E(eta*)            = {sigma_star:.4f}")
        print(f"  sigma_market             = {sigma_market:.4f}")
        print(f"  D/V                      = {dv_ratio:.4f}")
        print(f"  Residual |sigma_E - obs| = {abs_errors[best_idx]:.4f}")
        print(f"  Grid points evaluated    = {n_points}")
        print(f"{'='*55}\n")

    return eta_star, result_info