import numpy as np
from scipy import linalg as SciLA
from numpy import linalg as LA
from pauli import pauli_action, sigma_matrices
from binary_functions import Bas2Int, Int2Bas, Opp2Str
from tools import print_state, fidelity, graph_distance, periodic_boundary_distance, open_boundary_distance

# ------------------------------------------------- #
# Hamiltonian-related Functions

def Hpsi(H_, psi_):
    """
    Apply the Hamiltonian to a wavefunction.
    
    Parameters:
        H_ (list): Hamiltonian terms.
        psi_ (np.ndarray): Wavefunction.
    
    Returns:
        np.ndarray: Resulting wavefunction.
    """
    phi = np.zeros(psi_.shape, dtype=complex)
    for (A, h, imp, gmp) in H_:
        for m in np.where(np.abs(h) > 1e-8)[0]:
            phi += h[m] * gmp[m, imp[m, :]] * psi_[imp[m, :]]
    return phi


def Hmat(H_):
    """
    Construct the Hamiltonian matrix from its terms.
    
    Parameters:
        H_ (list): Hamiltonian terms.
    
    Returns:
        np.ndarray: Hamiltonian matrix.
    """
    N = H_[0][2].shape[1]
    Hm = np.zeros((N, N), dtype=complex)
    for i in range(N):
        ei = np.zeros(N, dtype=complex)
        ei[i] = 1.0
        Hm[:, i] = Hpsi(H_, ei)
    return Hm


def Hmoms(H_, psi_):
    """
    Compute the energy and variance of a wavefunction with the Hamiltonian.
    
    Parameters:
        H_ (list): Hamiltonian terms.
        psi_ (np.ndarray): Wavefunction.
    
    Returns:
        float: Energy expectation value.
        float: Variance.
    """
    phi_ = Hpsi(H_, psi_)
    ea = np.vdot(psi_, phi_)
    ev = np.vdot(phi_, phi_)
    return np.real(ea), np.real(ev - ea**2)


def print_Hamiltonian(H_):
    """
    Print the terms of the Hamiltonian in a readable format.
    
    Parameters:
        H_ (list): Hamiltonian terms.
    """
    for mu, (A, h, imp, gmp) in enumerate(H_):
        print(f"Term {mu}")
        print(f"Active qubits: {A}")
        print("Operators:")
        for m in np.where(np.abs(h) > 1e-8)[0]:
            print(f"{Opp2Str(Int2Bas(m, 4, len(A)))} {h[m]}")


def Hii(H_, i):
    """
    Compute the diagonal element of the Hamiltonian for a given state.
    
    Parameters:
        H_ (list): Hamiltonian terms.
        i (int): State index.
    
    Returns:
        float: Diagonal element value.
    """
    N = H_[0][2].shape[1]
    nbit = int(np.log2(N))
    hii = 0.0
    xi = Int2Bas(i, 2, nbit)
    for (A, h, imp, gmp) in H_:
        nact = len(A)
        for m in np.where(np.abs(h) > 1e-8)[0]:
            sm = Int2Bas(m, 4, nact)
            smx = [sigma_matrices[xi[A[w]], xi[A[w]], sm[w]] for w in range(nact)]
            hii += np.real(h[m] * np.prod(smx))
    return hii

# ------------------------------------------------- #
# Hamiltonian Generators

def Heisenberg_SR(nspin, R):
    """
    Generate the Heisenberg model with short-range interactions.
    
    Parameters:
        nspin (int): Number of spins.
        R (float): Interaction range.
    
    Returns:
        list: Hamiltonian terms.
    """
    H = []
    imax = nspin - 1 if nspin == 2 else nspin
    for i in range(imax):
        j = (i + 1) % nspin
        active = [k for k in range(nspin) if dpbc(i, k, nspin) < R or dpbc(j, k, nspin) < R]
        active = np.asarray(active)
        nact = len(active)
        h_alpha = np.zeros(4**nact)
        ii, jj = np.where(active == i)[0][0], np.where(active == j)[0][0]
        for alpha in range(1, 4):
            idx = [0] * nact
            idx[ii] = alpha
            idx[jj] = alpha
            h_alpha[Bas2Int(idx, 4)] = 1.0
        imap, gmap = pauli_action(active, nspin)
        H.append((active, h_alpha, imap, gmap))
    return H


def Heisenberg_LR(nspin, R):
    """
    Generate the Heisenberg model with long-range interactions.
    
    Parameters:
        nspin (int): Number of spins.
        R (float): Interaction range.
    
    Returns:
        list: Hamiltonian terms.
    """
    H = []
    for i in range(nspin):
        for j in range(i + 1, nspin):
            active = [k for k in range(nspin) if dobc(i, k, nspin) < R or dobc(j, k, nspin) < R]
            active = np.asarray(active)
            nact = len(active)
            h_alpha = np.zeros(4**nact)
            ii, jj = np.where(active == i)[0][0], np.where(active == j)[0][0]
            for alpha in range(1, 4):
                idx = [0] * nact
                idx[ii] = alpha
                idx[jj] = alpha
                h_alpha[Bas2Int(idx, 4)] = 1.0 / (dobc(i, j, nspin) + 1.0)
            imap, gmap = pauli_action(active, nspin)
            H.append((active, h_alpha, imap, gmap))
    return H


def MaxCut(graph, R):
    """
    Generate the MaxCut Hamiltonian for a given graph.
    
    Parameters:
        graph (tuple): Graph represented as (nodes, edges).
        R (float): Interaction range.
    
    Returns:
        list: Hamiltonian terms.
    """
    VV, EE = graph
    nbit = len(VV)
    H = []
    for (i, j) in EE:
        active = [k for k in range(nbit) if dgr(graph, i, k) < R or dgr(graph, j, k) < R]
        active = np.asarray(active)
        nact = len(active)
        h_alpha = np.zeros(4**nact)
        ii, jj = np.where(active == i)[0][0], np.where(active == j)[0][0]
        idx = [0] * nact
        h_alpha[Bas2Int(idx, 4)] = -0.5
        idx[ii], idx[jj] = 3, 3
        h_alpha[Bas2Int(idx, 4)] = 0.5
        imap, gmap = pauli_action(active, nbit)
        H.append((active, h_alpha, imap, gmap))
    return H


def Ising(nspin, R, psi):
    """
    Generate the Ising Hamiltonian for a 1D spin chain.
    
    Parameters:
        nspin (int): Number of spins.
        R (float): Interaction range.
        psi (float): Angle parameter for transverse field term.
    
    Returns:
        list: Hamiltonian terms.
    """
    H = []
    for i in range(nspin):
        j = (i + 1) % nspin
        active = [k for k in range(nspin) if dpbc(i, k, nspin) < R or dpbc(j, k, nspin) < R]
        active = np.asarray(active)
        nact = len(active)
        h_alpha = np.zeros(4**nact)

        # Interaction terms
        ii, jj = np.where(active == i)[0][0], np.where(active == j)[0][0]
        h_alpha[Bas2Int([3 if idx == ii else 0 for idx in range(nact)], 4)] = np.sin(psi) / 2.0
        h_alpha[Bas2Int([3 if idx == jj else 0 for idx in range(nact)], 4)] = np.sin(psi) / 2.0
        h_alpha[Bas2Int([1 if idx in {ii, jj} else 0 for idx in range(nact)], 4)] = np.cos(psi)

        imap, gmap = pauli_action(active, nspin)
        H.append((active, h_alpha, imap, gmap))
    return H


def Hubbard(norb, R, U):
    """
    Generate the Hubbard Hamiltonian for a lattice with a given interaction strength.
    
    Parameters:
        norb (int): Number of orbitals.
        R (float): Interaction range.
        U (float): On-site interaction strength.
    
    Returns:
        list: Hamiltonian terms.
    """
    H = []
    nspin = 2 * norb

    for i in range(norb - 1):
        print(f">>>>>>> sites {i}, {i + 1}")
        dij = np.asarray([min(np.abs(i - j), np.abs(i + 1 - j)) for j in range(norb)])
        idx = np.where(dij < R)[0]
        pmin, pmax = 2 * min(idx), 2 * max(idx) + 1
        active = list(range(pmin, pmax + 1))
        nact = len(active)
        h_alpha = np.zeros(4**nact)

        # Interaction terms
        for k in (i, i + 1):
            pk = 2 * k - pmin
            wk = 0.5 if k != 0 and k != norb - 1 else 1.0
            h_alpha[Bas2Int([3 if idx == pk else 0 for idx in range(nact)], 4)] += -U * wk / 4.0 + (U / 2) * wk / 2.0
            h_alpha[Bas2Int([3 if idx == pk + 1 else 0 for idx in range(nact)], 4)] += -U * wk / 4.0 + (U / 2) * wk / 2.0
            h_alpha[Bas2Int([3 if idx in {pk, pk + 1} else 0 for idx in range(nact)], 4)] += U * wk / 4.0

        # Kinetic energy terms
        for sigma in (0, 1):
            p = 2 * i + sigma - pmin
            q = p + 2
            h_alpha[Bas2Int([1 if idx == p else 3 if idx == p + 1 else 1 if idx == q else 0 for idx in range(nact)], 4)] = -0.5
            h_alpha[Bas2Int([2 if idx == p else 3 if idx == p + 1 else 2 if idx == q else 0 for idx in range(nact)], 4)] = -0.5

        imap, gmap = pauli_action(active, nspin)
        H.append((active, h_alpha, imap, gmap))
    return H


def H_molecule(line):
    """
    Generate the Hamiltonian for a molecule using data from an external file.
    
    Parameters:
        line (int): Line index in the data file to use.
    
    Returns:
        list: Hamiltonian terms.
    """
    # File path placeholder; adjust as needed
    data_path = '../../code_v4/h2.dat'
    V = np.loadtxt(data_path).T[:, line]
    nspin = 2
    H = []
    active = np.asarray([0, 1])
    nact = len(active)
    h_alpha = np.zeros(4**nact)

    h_alpha[Bas2Int([0, 0], 4)] = V[1]
    h_alpha[Bas2Int([3, 0], 4)] = V[2]
    h_alpha[Bas2Int([0, 3], 4)] = V[3]
    h_alpha[Bas2Int([3, 3], 4)] = V[4]
    h_alpha[Bas2Int([1, 1], 4)] = V[5]
    h_alpha[Bas2Int([2, 2], 4)] = V[6]

    imap, gmap = pauli_action(active, nspin)
    H.append((active, h_alpha, imap, gmap))
    return H

