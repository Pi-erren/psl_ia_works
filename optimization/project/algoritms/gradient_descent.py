import torch
import numpy as np
from scipy.linalg import norm  # Norm euclidienne (correspondant au produit scalaire)
from scipy.linalg import svdvals  # Décomposition en valeurs singulières
from problems.regression import RegPb


def sgd(
    problem,
    stepchoice=0,
    init_step=1,
    n_iter=1000,
    batch_size=1,
    grad_dist_criteria=None,
    verbose=False,
):
    """
    This function was highly inspired from Clément Royer: https://www.lamsade.dauphine.fr/%7Ecroyer/ensdocs/OAA/LabOAA-DescenteGradient-solutions.zip
    Implémentation de descente de gradient avec différentes tailles de pas.

    Entrées:
        w0: Point initial
        problem: Instance à minimiser
            problem.fun(x) Fonction objectif
            problem.grad(x) Gradient
            problem.lipgrad() Constante de Lipschitz pour le gradient
        grad_dist_criteria (float): stop algorithm when dist(w_{i+1} - w_{i}) <  grad_dist_criteria
        stepchoice: Choix de taille de pas
            0: Constante proportionnelle à 1/L (L constante de Lipschitz pour le gradient)
            a>0: Décroissante, set to 1/((k+1)**a)
        init_step: Taille de pas initiale
        n_iter: Maximum d'itérations
        verbose: Affichage d'informations à chaque itération

    Sorties:
        w_output: Dernier itéré de la méthode
        loss_val: Historique de valeurs de fonctions (tableau Numpy de taille n_iter+1)
        distits: Historique de distances à l'optimum cible (tableau Numpy de taille n_iter+1)
    """
    if stepchoice is None:
        return ValueError("Please select a stepchoice")

    # Utils variables
    loss_val = []
    k = 0

    # Algorithm loop
    while k < n_iter:
        # Update stepsize (e.g. learning rate)
        s = choose_step_size(stepchoice, init_step, k)

        # Proceed to a gradient descent step (w = w - lr * gradient)
        curr_loss = problem.step(s, batch_size)

        # Get curr loss for visualization
        loss_val.append(curr_loss)

        if verbose:
            print(f"k: {k} | loss: {curr_loss:.2e} | lr: {s:.2e}")
        k += 1
    return np.array(loss_val)


def choose_step_size(mode, init_step, k):
    # 1 - Définir la taille de pas s en fonction de k, L, step0 et g
    if mode == 0:
        # Taille de pas constante
        return init_step
    elif mode > 0:
        # Taille de pas décroissante
        return init_step / ((k + 1) ** mode)
