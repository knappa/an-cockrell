#!/usr/bin/env python3
# coding: utf-8

# # An-Cockrell model reimplementation

import math

import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File("run-statistics.hdf5", "r") as f:
    num_keys = len(f.keys())

    num_cols = 4
    num_rows = math.ceil(num_keys / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, sharex=True)

    for idx, key in enumerate(f.keys()):
        print(key)
        row, col = divmod(idx, num_cols)

        data = np.array(f[key])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        axs[row, col].plot(mean, color="black")
        axs[row, col].fill_between(
            range(len(mean)), mean - std, mean + std, alpha=0.35, color="black"
        )
        axs[row, col].set_title(str(key))
