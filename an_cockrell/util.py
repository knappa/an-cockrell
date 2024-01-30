#!python
# cython: language_level=3
import cython
import numpy as np

DTYPE = np.float64


def diffuse_molecule_field(molecule_field: np.ndarray, diffusion_constant: cython.double):
    # based on description at https://ccl.northwestern.edu/netlogo/docs/dict/diffuse.html

    row_max: cython.Py_ssize_t = molecule_field.shape[0]
    col_max: cython.Py_ssize_t = molecule_field.shape[1]

    result: np.ndarray = np.zeros((row_max, col_max), dtype=DTYPE)

    row_idx: cython.Py_ssize_t
    col_idx: cython.Py_ssize_t
    prev_row_idx: cython.Py_ssize_t
    next_row_idx: cython.Py_ssize_t
    prev_col_idx: cython.Py_ssize_t
    next_col_idx: cython.Py_ssize_t
    tmp: cython.double

    for row_idx in range(row_max):
        prev_row_idx = row_idx - 1 if row_idx > 0 else row_max - 1
        next_row_idx = row_idx + 1 if row_idx < row_max - 1 else 0
        for col_idx in range(1, col_max - 1):
            prev_col_idx = col_idx - 1 if col_idx > 0 else col_max - 1
            next_col_idx = col_idx + 1 if col_idx <= col_max - 1 else 0
            tmp = diffusion_constant * molecule_field[row_idx, col_idx] / 8.0
            result[prev_row_idx, prev_col_idx] += tmp
            result[prev_row_idx, col_idx] += tmp
            result[prev_row_idx, next_col_idx] += tmp
            result[row_idx, prev_col_idx] += tmp
            result[row_idx, col_idx] += (1 - diffusion_constant) * molecule_field[row_idx, col_idx]
            result[row_idx, next_col_idx] += tmp
            result[next_row_idx, prev_col_idx] += tmp
            result[next_row_idx, col_idx] += tmp
            result[next_row_idx, next_col_idx] += tmp

    molecule_field[:, :] = result
