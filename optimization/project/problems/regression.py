import numpy as np
from scipy.linalg import norm  # Norm euclidienne (correspondant au produit scalaire)
from scipy.linalg import svdvals  # Décomposition en valeurs singulières
from problems.problem import Problem


class RegPb(Problem):
    """
    Most of the code borrowed from Clément Royer: https://www.lamsade.dauphine.fr/%7Ecroyer/ensdocs/OAA/LabOAA-DescenteGradient-solutions.zip

    Classe de problèmes de régression avec modèles linéaires. L2 loss by default

    Attributs:
        X: Matrice de données (attributs)
        y: Vecteur de données (labels)
        n,d: Dimensions de X
        loss: Fonction de perte choisie
            'l2': Perte aux moindres carrés
            'logit': Perte logistique
        lambda_reg: Paramètre de régularisation
    """

    # Initialisation
    def __init__(self, X, y, w, lambda_reg=0):
        self.X = X
        self.y = y
        self.w = w
        self.w_list = [self.w]  # Keep track of parameters
        self.g_norm_list = []  # Keep track of gradients norm
        self.loss_vals = []
        self.n, self.d = X.shape
        self.lambda_reg = lambda_reg

    def step(self, s, batch_size):
        if batch_size is None:
            return ValueError("Regression step need a batch size in [1, n]")
        indices = np.random.choice(self.n, batch_size, replace=False)
        sg = np.mean([self.grad_i(i) for i in indices], axis=0)
        self.w = self.w - s * sg
        self.w_list.append(self.w)
        self.g_norm_list.append(norm(sg))
        self.loss_vals.append(self.loss())
        return self.loss()

    # Fonction objectif
    def loss(self):
        return (
            norm(self.X.dot(self.w) - self.y) ** 2 / (2.0 * self.n)
            + self.lambda_reg * norm(self.w) ** 2 / 2.0
        )

    # Gradient for one "data point"
    def grad_i(self, i):
        x_i = self.X[i]
        return (x_i.dot(self.w) - self.y[i]) * x_i + self.lambda_reg * self.w

    # Constante de Lipschitz pour le gradient
    def lipgrad(self):
        L = norm(self.X, ord=2) ** 2 / self.n + self.lambda_reg

    # Constante de convexité ''forte'' (potentiellement 0 si self.lambda_reg=0)
    def cvxval(self):
        s = svdvals(self.X)
        mu = min(s) ** 2 / self.n
        return mu + self.lambda_reg
