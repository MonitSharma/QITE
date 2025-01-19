import numpy as np
from numpy import linalg as LA
from scipy import linalg as SciLA
from scipy.optimize import minimize
from pyscf.lib import davidson



from pauli import pauli_action, sigma_matrices
from binary_functions import Bas2Int, Int2Bas
from tools import print_state, fidelity, dump_state, read_state, dump_lanczos_results
from hamiltonian import Hmat, Hmoms, Hpsi

# -------------------------------------------------
# Exponential Evolution

def ExpitH(H_, psi_, dT):
    """
    Perform real-time evolution of the wavefunction.

    Parameters:
        H_ (list): Hamiltonian terms.
        psi_ (np.ndarray): Initial wavefunction.
        dT (float): Time step for evolution.

    Returns:
        tuple: Updated wavefunction after evolution and its norm.
    """
    phi = psi_.copy()
    chi = psi_.copy()
    i = 0
    while LA.norm(chi) > 1e-6:
        chi = (1j * dT / float(i + 1)) * Hpsi(H_, chi)
        phi += chi
        i += 1
    nu = LA.norm(phi)
    return phi / nu, nu

# -------------------------------------------------
# Real-Time Evolution (RTE)

def RTE(H_, dT, Tmax, lanczos=False, psi0=None):
    """
    Perform Real-Time Evolution (RTE) using the given Hamiltonian.

    Parameters:
        H_ (list): Hamiltonian terms.
        dT (float): Time step for evolution.
        Tmax (float): Maximum evolution time.
        lanczos (bool): Whether to use Lanczos for optimization. Default is False.
        psi0 (np.ndarray): Initial wavefunction. If None, uses the ground state.

    Outputs:
        Writes results to 'RTE_davidson.out' and saves vectors to 'qlanz.vecs'.
    """
    # --- Diagonalization ---
    N = H_[0][2].shape[1]
    nbit = int(np.log2(N))
    hdiag = np.zeros(N, dtype=complex)
    for i in range(N):
        xi = Int2Bas(i, 2, nbit)
        for (A, h, imp, gmp) in H_:
            nact = len(A)
            for m in np.where(np.abs(h) > 1e-8)[0]:
                sm = Int2Bas(m, 4, nact)
                smx = [sigma_matrices[xi[A[w]], xi[A[w]], sm[w]] for w in range(nact)]
                hdiag[i] += h[m] * np.prod(smx)
        if i % 1000 == 0:
            print(f"Processing state {i}/{N}: Energy = {hdiag[i]}")

    precond = lambda x, e, *args: x / (hdiag - e + 1e-4)

    def hop(c_):
        return Hpsi(H_, c_)

    epsm0, Um0 = davidson(hop, psi0, precond)

    # --- Output ground-state energy ---
    with open('RTE_davidson.out', 'w') as fout:
        fout.write(f"Ground-state energy: {epsm0:.6f}\n")

        # --- Initialize the wavefunction ---
        if psi0 is None:
            i0 = np.argmin(hdiag)
            psi0 = np.zeros(N, dtype=complex)
            psi0[i0] = 1.0

        # --- Real-time evolution ---
        bra_RTE = psi0.copy()
        braH_RTE = Hpsi(H_, psi0.copy())
        ket_RTE = psi0.copy()

        nbeta = int(Tmax / dT) + 1
        hvect_LANZ = np.zeros(nbeta + 1, dtype=complex)
        svect_LANZ = np.zeros(nbeta + 1, dtype=complex)

        fout.write("RTE Progression:\n")
        for ib in range(nbeta):
            hvect_LANZ[ib] = np.vdot(braH_RTE, ket_RTE)
            svect_LANZ[ib] = np.vdot(bra_RTE, ket_RTE)
            ket_RTE = ExpitH(H_, ket_RTE, dT)[0]
            fout.write(f"Time: {ib * dT:.6f}, Energy: {hvect_LANZ[ib].real:.6f}\n")
            print(f"Step {ib}, Energy: {hvect_LANZ[ib]}")

        # Save the results
        dump_lanz_rte(hvect_LANZ[:nbeta], svect_LANZ[:nbeta], 'qlanz.vecs')

# -------------------------------------------------
