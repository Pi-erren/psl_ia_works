import matplotlib.pyplot as plt
import torch
import numpy as np


def frobenius_norm(U, V, X, lambda_reg=0):
    UV = U @ V.T
    n = UV.shape[0] * UV.shape[1]
    loss = (1 / (2 * n)) * (torch.norm(UV - X, p="fro") ** 2)
    reg = (lambda_reg / 2) * (
        (torch.norm(U, p="fro") ** 2 + torch.norm(V, p="fro") ** 2)
    )
    return loss + reg


def visualize(data: dict, subject: str, window_size=None):
    """
    Args:
        - data: dictionnary with plot labels as keys and list as values
    """
    plt.figure(figsize=(7, 5))
    for label, values in data.items():
        if window_size is not None:
            values = np.convolve(
                values, np.ones(window_size) / window_size, mode="valid"
            )
        plt.plot(values, label=label, lw=2)
    plt.title(f"Convergence in terms of {subject}", fontsize=16)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel(subject, fontsize=14)
    plt.legend()


def create_r_rank_mat(dim, r):
    A = torch.zeros(dim)
    for i in range(r):
        a_i = torch.randn(dim[0], r)
        b_i = torch.randn(dim[1], r)
        A += a_i @ b_i.T
    return A
