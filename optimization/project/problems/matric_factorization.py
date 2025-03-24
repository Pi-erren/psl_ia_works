import numpy as np
from scipy.linalg import norm  # Norm euclidienne (correspondant au produit scalaire)
from scipy.linalg import svdvals  # Décomposition en valeurs singulières
import torch
from torch import nn

from problems.problem import Problem


class MatFactPb(Problem):
    """ """

    # Initialisation
    def __init__(self, n1, n2, X, r=1, lambda_reg=0):
        self.w = torch.randn(n1 * r + n2 * r, requires_grad=True)
        self.X = X
        self.n1 = n1
        self.n2 = n2
        self.r = r
        self.lambda_reg = lambda_reg
        self.loss_vals = []

    # Loss using Frobenius norm
    def loss(self):
        U, V = self.w[: self.n1 * self.r].view(self.n1, self.r), self.w[
            self.n1 * self.r :
        ].view(self.n2, self.r)

        loss = (0.5 / (self.n1 * self.n2)) * torch.norm(U @ V.T - self.X, p="fro") ** 2
        reg = (self.lambda_reg / 2) * (
            (torch.norm(U, p="fro") ** 2 + torch.norm(V, p="fro") ** 2)
        )
        return loss + reg

    def batch_loss(self, batch_size):
        U, V = self.w[: self.n1 * self.r].view(self.n1, self.r), self.w[
            self.n1 * self.r :
        ].view(self.n2, self.r)
        u_indexes = torch.randint(0, self.n1, (batch_size,))
        v_indexes = torch.randint(0, self.n2, (batch_size,))

        loss = (0.5 / (batch_size * batch_size)) * torch.norm(
            U[u_indexes] @ V[v_indexes].T - self.X[u_indexes][:, v_indexes],
            p="fro",
        ) ** 2
        reg = (self.lambda_reg / 2) * (
            (
                torch.norm(U[u_indexes], p="fro") ** 2
                + torch.norm(V[v_indexes].T, p="fro") ** 2
            )
        )
        return loss + reg

    # Calcul de gradient
    def step(self, lr, batch_size=None):

        # Compte loss
        if batch_size is None:
            loss = self.loss()
        else:
            loss = self.batch_loss(batch_size=batch_size)

        # Compute gradient
        loss.backward()
        self.loss_vals.append(loss.item())

        # Update parameter with algorithm step formula
        with torch.no_grad():
            self.w -= lr * self.w.grad
            self.w.grad.zero_()

        return loss.item()

    def UV_xz_distance(self, xzT):
        U, V = self.w[: self.n1 * self.r].view(self.n1, self.r), self.w[
            self.n1 * self.r :
        ].view(self.n2, self.r)
        return torch.norm(U @ V.T - xzT, p="fro")
