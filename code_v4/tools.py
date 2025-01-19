import numpy as np
import pickle
from binary_functions import Int2Bas

def fidelity(psi1, psi2):
    """
    Calculate the fidelity between two quantum states.
    Parameters:
        psi1, psi2: numpy arrays
            Complex vectors representing quantum states.
    Returns:
        float: Fidelity (absolute value of the dot product).
    """
    return np.abs(np.vdot(psi1, psi2))

def print_state(state, n_bits, output_file):
    """
    Print the quantum state in a human-readable format.
    Parameters:
        state: numpy array
            Quantum state vector.
        n_bits: int
            Number of qubits.
        output_file: file object
            File to write the state to.
    """
    for i in range(state.shape[0]):
        if np.abs(state[i]) > 1e-4:
            binary_representation = Int2Bas(i, 2, n_bits)
            output_file.write("".join(map(str, binary_representation)))
            output_file.write(f" {np.real(state[i]):.12f} {np.imag(state[i]):.12f} I \n")

def dump_state(state, n_bits, file_name):
    """
    Dump the quantum state to a file, only including significant amplitudes.
    Parameters:
        state: numpy array
            Quantum state vector.
        n_bits: int
            Number of qubits.
        file_name: str
            Path to the output file.
    """
    with open(file_name, 'w') as output_file:
        for index in np.where(np.abs(state) > 1e-4)[0]:
            output_file.write(f"{index} {np.real(state[index]):.12f} {np.imag(state[index]):.12f}\n")

def read_state(n_bits, file_name):
    """
    Read a quantum state from a file.
    Parameters:
        n_bits: int
            Number of qubits.
        file_name: str
            Path to the input file.
    Returns:
        numpy array: Reconstructed quantum state vector.
    """
    state = np.zeros(2**n_bits, dtype=complex)
    data = np.loadtxt(file_name)
    indices = data[:, 0].astype(int)
    state[indices] = data[:, 1] + 1j * data[:, 2]
    return state

def dump_lanczos_vectors(hamiltonian_values, scalar_values, file_name):
    """
    Dump Lanczos vectors to a file.
    Parameters:
        hamiltonian_values: list of floats
            Eigenvalues of the Hamiltonian.
        scalar_values: list of floats
            Scalar values for Lanczos vectors.
        file_name: str
            Path to the output file.
    """
    with open(file_name, 'w') as output_file:
        for i, (h, s) in enumerate(zip(hamiltonian_values, scalar_values)):
            output_file.write(f"{i} {h:.12f} {s:.12f}\n")

def dump_lanczos_results(hamiltonian_values, scalar_values, file_name):
    """
    Dump Lanczos eigenvalues and vectors with complex components.
    Parameters:
        hamiltonian_values: list of complex
            Complex eigenvalues of the Hamiltonian.
        scalar_values: list of complex
            Complex scalar values for Lanczos vectors.
        file_name: str
            Path to the output file.
    """
    with open(file_name, 'w') as output_file:
        for i, (h, s) in enumerate(zip(hamiltonian_values, scalar_values)):
            output_file.write(
                f"{i} {np.real(h):.8f} {np.imag(h):.8f} {np.real(s):.8f} {np.imag(s):.8f}\n"
            )

# ------------------------------------------------- #

def periodic_boundary_distance(a, b, n):
    """
    Calculate the minimum distance on a periodic boundary condition.
    Parameters:
        a, b: int
            Positions on the boundary.
        n: int
            Total length of the boundary.
    Returns:
        int: Minimum distance.
    """
    return min((a - b) % n, (b - a) % n)

def open_boundary_distance(a, b, n):
    """
    Calculate the distance on an open boundary condition.
    Parameters:
        a, b: int
            Positions on the boundary.
        n: int
            Total length of the boundary.
    Returns:
        int: Absolute distance.
    """
    return abs(a - b)

def graph_distance(graph, start, end):
    """
    Calculate the shortest path distance in a graph using breadth-first search.
    Parameters:
        graph: tuple
            A graph represented as (nodes, edges).
        start, end: int
            Start and end nodes.
    Returns:
        int: Shortest path length.
    """
    nodes, edges = graph
    n_bits = len(nodes)
    
    if start == end:
        return 0

    paths = [[start]]
    while paths:
        new_paths = []
        for path in paths:
            last_node = path[-1]
            for edge in edges:
                if last_node in edge:
                    next_node = edge[1] if edge[0] == last_node else edge[0]
                    if next_node == end:
                        return len(path)
                    if next_node not in path:
                        new_paths.append(path + [next_node])
        paths = new_paths
    return float('inf')
