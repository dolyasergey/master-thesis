#helper functions

# imports
import pandas as pd
import numpy as np

# parameters import
def load_params_csv(path):
    df = pd.read_csv(path)

    params = dict(zip(df['par'], df['value']))

    return params

# create time grid
def make_time_grid(T, n_steps):
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    return t, dt

# construct correlation matrix
def rho_matrix_construct(n, rhos):
    corr_matrix = np.eye(n)
    indices = np.triu_indices(n, k=1)
    corr_matrix[indices] = rhos
    corr_matrix[(indices[1], indices[0])] = rhos
    return corr_matrix





