import numpy as np
from scipy import optimize as opt
from pauli import pauli_action, sigma_matrices
from binary_functions import Bas2Int, Int2Bas
from numpy import linalg as LA
from scipy import linalg as SciLA
from tools import print_state, fidelity, graph_distance

# -------------------------------------------------
# Mean-Field State and Energy Calculations

def mf_state(theta):
    """
    Generate the mean-field state given a set of angles.

    Parameters:
        theta (np.ndarray): Array of angles defining the state.

    Returns:
        np.ndarray: The mean-field wavefunction as a complex vector.
    """
    nbit = theta.shape[0]
    chi = np.zeros((nbit, 2))
    for i in range(nbit):
        chi[i, 0] = np.cos(theta[i])
        chi[i, 1] = np.sin(theta[i])
    
    N = 2**nbit
    phi = np.zeros(N, dtype=complex)
    for i in range(N):
        x = Int2Bas(i, 2, nbit)
        phi[i] = np.prod([chi[k, x[k]] for k in range(nbit)])
    return phi


def mf_energy(theta, H_):
    """
    Compute the mean-field energy for a given set of angles.

    Parameters:
        theta (np.ndarray): Array of angles defining the state.
        H_ (list): Hamiltonian terms.

    Returns:
        float: The mean-field energy.
    """
    nbit = theta.shape[0]
    chi = np.zeros((nbit, 2))
    for i in range(nbit):
        chi[i, 0] = np.cos(theta[i])
        chi[i, 1] = np.sin(theta[i])

    # Compute the expectation values of Pauli matrices
    xjm = np.einsum("ja,abc,jb->jc", chi, sigma_matrices, chi)
    xjm = np.real(xjm)
    
    ea = 0.0
    for (A, h, imp, gmp) in H_:
        nact = len(A)
        for m in np.where(np.abs(h) > 1e-8)[0]:
            xm = Int2Bas(m, 4, nact)
            ea += h[m] * np.prod([xjm[A[k], xm[k]] for k in range(nact)])
    return ea


def mf_solution(theta0, H_):
    """
    Solve the mean-field problem using optimization.

    Parameters:
        theta0 (np.ndarray): Initial guess for the angles.
        H_ (list): Hamiltonian terms.

    Returns:
        tuple: Optimized angles and the corresponding mean-field energy.
    """
    res = opt.minimize(mf_energy, theta0, args=H_, method="SLSQP")
    return res.x, mf_energy(res.x, H_)

# -------------------------------------------------
# Homogeneous Mean-Field State and Energy Calculations

def hom_mf_state(theta, nbit):
    """
    Generate a homogeneous mean-field state for a single angle.

    Parameters:
        theta (float): Angle defining the homogeneous state.
        nbit (int): Number of qubits.

    Returns:
        np.ndarray: The homogeneous mean-field wavefunction.
    """
    chi = np.array([np.cos(theta), np.sin(theta)])
    N = 2**nbit
    phi = np.zeros(N, dtype=complex)
    for i in range(N):
        x = Int2Bas(i, 2, nbit)
        phi[i] = np.prod([chi[x[k]] for k in range(nbit)])
    return phi


def hom_mf_energy(theta, nbit, H_):
    """
    Compute the homogeneous mean-field energy for a single angle.

    Parameters:
        theta (float): Angle defining the homogeneous state.
        nbit (int): Number of qubits.
        H_ (list): Hamiltonian terms.

    Returns:
        float: The homogeneous mean-field energy.
    """
    chi = np.array([np.cos(theta), np.sin(theta)])
    xjm = np.einsum("a,abc,b->c", chi, sigma_matrices, chi)
    xjm = np.real(xjm)
    
    ea = 0.0
    for (A, h, imp, gmp) in H_:
        nact = len(A)
        for m in np.where(np.abs(h) > 1e-8)[0]:
            xm = Int2Bas(m, 4, nact)
            ea += h[m] * np.prod([xjm[xm[k]] for k in range(nact)])
    return ea


def hom_mf_solution(theta0, nbit, H_):
    """
    Solve the homogeneous mean-field problem using optimization.

    Parameters:
        theta0 (float): Initial guess for the angle.
        nbit (int): Number of qubits.
        H_ (list): Hamiltonian terms.

    Returns:
        tuple: Optimized angle and the corresponding homogeneous mean-field energy.
    """
    res = opt.minimize(hom_mf_energy, theta0, args=(nbit, H_), method="SLSQP")
    return res.x, hom_mf_energy(res.x, nbit, H_)

# -------------------------------------------------
