import numpy as np
from scipy import linalg
from binary_functions import Int2Bas, Bas2Int, Opp2Str, Psi2Str, Lst2Str

# ---------------------------------------------------------- #
# Predefined Constants and Pauli Matrices

# Define Pauli product rules
pauli_product = [
    np.zeros((4, 4), dtype=int),
    np.zeros((4, 4), dtype=complex)
]

# Pauli product indices
pauli_product[0][0, :] = [0, 1, 2, 3]
pauli_product[0][1, :] = [1, 0, 3, 2]
pauli_product[0][2, :] = [2, 3, 0, 1]
pauli_product[0][3, :] = [3, 2, 1, 0]

# Pauli product coefficients
pauli_product[1][0, :] = [1, 1, 1, 1]
pauli_product[1][1, :] = [1, 1, 1j, -1j]
pauli_product[1][2, :] = [1, -1j, 1, 1j]
pauli_product[1][3, :] = [1, 1j, -1j, 1]

# Define Pauli matrices
sigma_matrices = np.zeros((2, 2, 4), dtype=complex)
for i in range(2):
    j = (i + 1) % 2
    sigma_matrices[i, i, 0] = 1.0  # Identity
    sigma_matrices[i, j, 1] = 1.0  # Pauli-X
    sigma_matrices[i, j, 2] = 1.0j * (-1.0) ** (i + 1.0)  # Pauli-Y
    sigma_matrices[i, i, 3] = (-1.0) ** i  # Pauli-Z

# Helper functions using lambda for specific conditions
d12 = lambda t: 1 if t % 3 > 0 else 0
d12f = np.vectorize(d12)
d2 = lambda t: 1 if t == 2 else 0
d2f = np.vectorize(d2)
d23 = lambda t: 1 if t > 1 else 0
d23f = np.vectorize(d23)

# ---------------------------------------------------------- #

def computational_basis(n_bits):
    """
    Display the computational basis states for n_bits qubits.
    
    Parameters:
        n_bits (int): Number of qubits.
    """
    total_states = 2 ** n_bits
    for i in range(total_states):
        print(i, Psi2Str(Int2Bas(i, 2, n_bits)))

def pauli_basis(n_bits):
    """
    Display the Pauli basis states for n_bits qubits.
    
    Parameters:
        n_bits (int): Number of qubits.
    """
    total_operators = 4 ** n_bits
    for i in range(total_operators):
        print(i, Opp2Str(Int2Bas(i, 4, n_bits)))

# ---------------------------------------------------------- #

def pauli_action(active_indices, n_bits, verbose=False):
    """
    Compute the action of Pauli operators on qubits.
    
    Parameters:
        active_indices (list): List of active qubit indices.
        n_bits (int): Total number of qubits.
        verbose (bool): Whether to print intermediate steps for debugging.
    
    Returns:
        np.ndarray: Index mapping of states after Pauli action.
        np.ndarray: Coefficients resulting from Pauli action.
    """
    num_active = len(active_indices)
    total_states = 2 ** n_bits
    total_pauli_operators = 4 ** num_active

    # Precompute dot product for binary index conversion
    dot_product = [2 ** (n_bits - 1 - i) for i in range(n_bits)]
    indices_matrix = np.zeros((total_pauli_operators, total_states), dtype=int)
    coefficients_matrix = np.zeros((total_pauli_operators, total_states), dtype=complex) + 1

    # Generate state vector for Pauli operators
    pauli_vectors = np.zeros((total_pauli_operators, n_bits), dtype=int)
    for mu in range(total_pauli_operators):
        pauli_vectors[mu, active_indices] = Int2Bas(mu, 4, num_active)

    # Define vectorized transformations
    xy_vectors = d12f(pauli_vectors)  # Tracks X and Y components
    y_vectors = d2f(pauli_vectors)   # Tracks Y components only
    yz_vectors = d23f(pauli_vectors)  # Tracks Y and Z components
    y_count = np.einsum('ab->a', y_vectors)

    # Create binary representations for computational basis
    binary_states = np.zeros((total_states, n_bits), dtype=int)
    for state_index in range(total_states):
        binary_states[state_index, :] = np.asarray(Int2Bas(state_index, 2, n_bits))

    # Compute coefficients for each state
    coefficients_matrix = np.einsum('am,bm->ba', binary_states, yz_vectors) + 0j
    coefficients_matrix[:, :] = (-1) ** coefficients_matrix[:, :]
    for mu in range(total_pauli_operators):
        coefficients_matrix[mu, :] *= 1j ** y_count[mu]
        new_states = (binary_states + xy_vectors[mu, :]) % 2
        indices_matrix[mu, :] = np.einsum('a,ba->b', dot_product, new_states)

    return indices_matrix, coefficients_matrix
