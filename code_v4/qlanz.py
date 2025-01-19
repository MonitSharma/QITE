import numpy as np
import sys
from numpy import linalg as LA
from scipy import linalg as SciLA

# --------- Utility Functions ---------

def read_lanczos_vectors(file_name):
    """
    Read Lanczos vectors from a file.
    
    Parameters:
        file_name (str): Path to the file containing Lanczos vectors.
    
    Returns:
        int: Number of beta values.
        np.ndarray: Array of Hamiltonian values (hv).
        np.ndarray: Array of scalar values (sv).
    """
    data = np.loadtxt(file_name)
    num_beta = data.shape[0]
    hv = data[:, 1]
    sv = data[:, 2]
    return num_beta, hv, sv


def rarefied_sampling(h_matrix, s_matrix, indices):
    """
    Perform rarefied sampling by selecting a subset of rows and columns.
    
    Parameters:
        h_matrix (np.ndarray): Hamiltonian matrix.
        s_matrix (np.ndarray): Overlap matrix.
        indices (list): List of indices to select.
    
    Returns:
        int: Number of selected variables.
        np.ndarray: Subsampled Hamiltonian matrix.
        np.ndarray: Subsampled overlap matrix.
    """
    s_matrix = s_matrix[np.ix_(indices, indices)]
    h_matrix = h_matrix[np.ix_(indices, indices)]
    num_vars = s_matrix.shape[0]
    return num_vars, h_matrix, s_matrix


def diagonal_regularization(h_matrix, s_matrix, factor=1):
    """
    Apply diagonal regularization to the matrices.
    
    Parameters:
        h_matrix (np.ndarray): Hamiltonian matrix.
        s_matrix (np.ndarray): Overlap matrix.
        factor (float): Regularization factor.
    
    Returns:
        np.ndarray: Regularized Hamiltonian matrix.
        np.ndarray: Regularized overlap matrix.
    """
    num_vars = h_matrix.shape[0]
    for i in range(num_vars):
        h_matrix[i, i] *= factor
        s_matrix[i, i] *= factor
    return h_matrix, s_matrix

# --------- Lanczos Functions ---------

def lanczos_kernel(h_matrix, s_matrix, indices, reference_energy):
    """
    Lanczos kernel for solving the eigenvalue problem.
    
    Parameters:
        h_matrix (np.ndarray): Hamiltonian matrix.
        s_matrix (np.ndarray): Overlap matrix.
        indices (list): Indices for rarefied sampling.
        reference_energy (float): Reference energy value.
    
    Returns:
        float: Minimum eigenvalue.
    """
    num_vars, h_sub, s_sub = rarefied_sampling(h_matrix, s_matrix, indices)
    eigenvalues, eigenvectors = SciLA.eigh(s_sub)
    
    valid_indices = np.where(eigenvalues > 1e-14)[0]
    s_sub = np.dot(eigenvectors.T, np.dot(s_sub, eigenvectors))
    h_sub = np.dot(eigenvectors.T, np.dot(h_sub, eigenvectors))
    
    s_sub = s_sub[np.ix_(valid_indices, valid_indices)]
    h_sub = h_sub[np.ix_(valid_indices, valid_indices)]
    
    eigenvalues, _ = SciLA.eigh(h_sub, s_sub)
    return np.min(np.real(eigenvalues))

# --------- Lanczos QITE ---------

def lanczos_qite(file_name, max_vars, delta_beta, delta_s, num_neighbors=None):
    """
    Perform Quantum Imaginary Time Evolution (QITE) using Lanczos method.
    
    Parameters:
        file_name (str): Path to the file containing Lanczos vectors.
        max_vars (int): Maximum number of variables.
        delta_beta (float): Step size for beta updates.
        delta_s (float): Threshold for overlap matrix sparsity.
        num_neighbors (int, optional): Number of neighbors to consider. Default is None.
    """
    num_beta, h_values, s_values = read_lanczos_vectors(file_name)
    energy_ref = (h_values[0] + h_values[num_beta - 1]) / 2

    for i in range(num_beta):
        s_values[i] += delta_beta * i * energy_ref
        h_values[i] -= energy_ref

    indices_list = list(range(0, num_beta, 2))
    num_vars = len(indices_list)
    h_matrix = np.zeros((num_vars, num_vars))
    s_matrix = np.zeros((num_vars, num_vars))

    for i in indices_list:
        for j in indices_list:
            mid_index = (i + j) // 2
            s_matrix[i // 2, j // 2] = np.exp(2 * s_values[mid_index] - s_values[i] - s_values[j])
            h_matrix[i // 2, j // 2] = h_values[mid_index] * s_matrix[i // 2, j // 2]

    s_matrix = s_matrix[:max_vars, :max_vars]
    h_matrix = h_matrix[:max_vars, :max_vars]

    selected_indices = [0]
    i = 0
    while i < max_vars and len(selected_indices) < max_vars:
        for j in range(i + 1, max_vars):
            if s_matrix[i, j] < delta_s:
                selected_indices.append(j)
                break
        i = selected_indices[-1]

    selected_indices.append(max_vars - 1)
    estimated_energy = lanczos_kernel(h_matrix, s_matrix, selected_indices, 0)
    final_energy = estimated_energy + energy_ref

    print(f"Max Variables: {max_vars}, Delta S: {delta_s}, Energy: {final_energy}")
    sys.stdout.flush()
