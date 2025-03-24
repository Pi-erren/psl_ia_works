import numpy as np
from scipy.linalg import norm  # Norm euclidienne (correspondant au produit scalaire)
from scipy.linalg import svdvals  # Décomposition en valeurs singulières
import torch
from torch import nn

from problems.problem import Problem

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
        self.loss_vals = []

    # Loss using Frobenius norm
    def loss(self):
        U, V = self.w[: self.n1 * self.r].view(self.n1, self.r), self.w[
            self.n1 * self.r :
        ].view(self.n2, self.r)
        return (0.5 / (self.n1 * self.n2)) * torch.norm(U @ V.T - self.X, p="fro") ** 2

    def batch_loss(self, batch_size):
        U, V = self.w[: self.n1 * self.r].view(self.n1, self.r), self.w[
            self.n1 * self.r :
        ].view(self.n2, self.r)
        u_indexes = torch.randint(0, self.n1, (batch_size,))
        v_indexes = torch.randint(0, self.n2, (batch_size,))
        return (0.5 / (batch_size * batch_size)) * torch.norm(
            U[u_indexes] @ V[v_indexes].T - self.X[u_indexes][:, v_indexes],
            p="fro",
        ) ** 2

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


class MatFactPb(Problem):
    """ """

    # Initialisation
    def __init__(self, n1, n2, X, lambda_reg=0):
        self.w = torch.randn(n1 + n2, requires_grad=True)
        self.X = X
        self.n1 = n1
        self.n2 = n2
        self.loss_vals = []

    # Loss using Frobenius norm
    def loss(self):
        u, v = self.w[: self.n1], self.w[self.n1 :]
        return (0.5 / (self.n1 * self.n2)) * torch.norm(
            torch.outer(u, v) - self.X, p="fro"
        ) ** 2

    def batch_loss(self, batch_size):
        u, v = self.w[: self.n1], self.w[self.n1 :]
        u_indexes = torch.randint(0, self.n1, (batch_size,))
        v_indexes = torch.randint(0, self.n2, (batch_size,))
        return (0.5 / (batch_size * batch_size)) * torch.norm(
            torch.outer(u[u_indexes], v[v_indexes]) - self.X[u_indexes][:, v_indexes],
            p="fro",
        ) ** 2

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


class MatFactPb(Problem):
    """
    Matrix factorization problem class.
    For the sake of simplicity, we use pytorch to differentiate the frobenius norm.

    Attributs:
        u: vercteur de données (attributs)
        v: Vecteur de données (labels)
        X
        n,d: Dimensions de X
        loss: Fonction de perte choisie
            'l2': Perte aux moindres carrés
            'logit': Perte logistique
        lambda_reg: Paramètre de régularisation
    """

    # Initialisation
    def __init__(self, n1, n2, r, X, lambda_reg=0):
        self.UV = torch.randn(n1 * r + n2 * r, requires_grad=True)
        self.X = torch.tensor(X, requires_grad=False)

        self.UV_grad_norm_list = []
        self.loss_vals = []
        self.lambda_reg = lambda_reg
        self.n1, self.n2 = n1, n2
        self.r = r

    # Loss using Frobenius norm
    def loss(self):
        UV = self.U @ self.V.T
        n = UV.shape[0] * UV.shape[1]
        loss = (1 / (2 * n)) * (torch.norm(UV - self.X, p="fro") ** 2)
        reg = (self.lambda_reg / 2) * (
            (torch.norm(self.U, p="fro") ** 2 + torch.norm(self.V, p="fro") ** 2)
        )
        return loss + reg

    def batch_loss(self, batch_size):
        # Draw batch for each parameter u and v
        U_index = torch.randint(0, self.n1, (batch_size,))
        V_index = torch.randint(0, self.n2, (batch_size,))
        U_batch = self.UV[: self.n1 * self.r][U_index]
        V_vatch = self.UV[: self.n2 * self.r][V_index]
        X_batch = self.X[U_index][:, V_index]

        # Compute objective function
        loss = (0.5 / (batch_size * batch_size)) * (
            torch.norm(torch.outer(U_batch, V_vatch.T) - X_batch, p="fro") ** 2
        )
        reg = (self.lambda_reg / 2) * (
            (torch.norm(U_batch, p="fro") ** 2 + torch.norm(V_vatch, p="fro") ** 2)
        )
        return loss + reg

    # Calcul de gradient
    def step(self, lr, batch_size):
        # Compute loss
        if not (batch_size is None):
            loss = self.batch_loss(batch_size)
        else:
            loss = self.loss()
        self.loss_vals.append(loss.item())

        # Compute gradient
        loss.backward()

        # Update parameter
        with torch.no_grad():
            UVgrad = self.UV.grad
            self.UV -= lr * UVgrad
        # Save gradients norms
        self.UV_grad_norm_list.append(torch.norm(UVgrad))

        # Reset gradient for next iter
        self.UV.grad.zero_()
