#!python
#cython: language_level=3

import numpy as np

DTYPE = np.float64

def diffuse_molecule_field(
        molecule_field: np.ndarray, float diffusion_constant
):
    # based on description at https://ccl.northwestern.edu/netlogo/docs/dict/diffuse.html

    cdef Py_ssize_t row_max = molecule_field.shape[0]
    cdef Py_ssize_t col_max = molecule_field.shape[1]

    result = np.zeros((row_max, col_max), dtype=DTYPE)

    cdef Py_ssize_t row_idx, col_idx, prev_row_idx, next_row_idx, prev_col_idx, next_col_idx
    cdef float tmp

    for row_idx in range(row_max):
        prev_row_idx = row_idx - 1 if row_idx > 0 else row_max - 1
        next_row_idx = row_idx + 1 if row_idx < row_max - 1 else 0
        for col_idx in range(1,col_max-1):
            prev_col_idx = col_idx - 1 if col_idx > 0 else col_max - 1
            next_col_idx = col_idx + 1 if col_idx <= col_max - 1 else 0
            tmp = diffusion_constant * molecule_field[row_idx,col_idx] / 8.0
            result[prev_row_idx, prev_col_idx] += tmp
            result[prev_row_idx, col_idx] += tmp
            result[prev_row_idx, next_col_idx] += tmp
            result[row_idx, prev_col_idx] += tmp
            result[row_idx, col_idx] += (1-diffusion_constant) * molecule_field[row_idx,col_idx]
            result[row_idx, next_col_idx] += tmp
            result[next_row_idx, prev_col_idx] += tmp
            result[next_row_idx, col_idx] += tmp
            result[next_row_idx, next_col_idx] += tmp

    molecule_field[:,:] = result
