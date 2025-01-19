import numpy as np
from scipy import linalg as SciLA
from numpy import linalg as LA
from binary_functions import Bas2Int, Int2Bas
from tools import print_state, fidelity, dump_state, read_state, dump_lanczos_vectors
from hamiltonian import Hmat, Hmoms, Hpsi, Hii


def ITE_FCI(H_, db, bmax, psi0=None, omega=None):
    """
    Perform Imaginary Time Evolution (ITE) using Full Configuration Interaction (FCI).

    Parameters:
        H_ (np.ndarray): Hamiltonian in matrix form.
        db (float): Time step for imaginary time evolution.
        bmax (float): Maximum imaginary time.
        psi0 (np.ndarray, optional): Initial wavefunction. Defaults to None.
        omega (np.ndarray, optional): Target states for fidelity calculation. Defaults to None.
    """
    Hm = Hmat(H_)
    N = Hm.shape[0]
    nbit = int(np.log2(N))
    eps, U = SciLA.eigh(Hm)
    m0 = np.argmin(eps)
    zeta = np.exp(-db * (eps - eps[m0]))

    with open("ITE_FCI.out", "w") as fout:
        fout.write(f"FCI ground-state energy: {eps[m0]:.6f}\n")
        fout.write("FCI ground-state wavefunction:\n")
        print_state(U[:, m0], nbit, fout)

        psi_FCI = np.zeros(N, dtype=complex) if psi0 is None else psi0.copy()
        if psi0 is None:
            psi_FCI[np.argmin(np.diag(Hm))] = 1.0

        nbeta = int(bmax / db) + 1
        fout.write("FCI ITE Progression:\n")
        for ib in range(nbeta):
            ea, ev = Hmoms(H_, psi_FCI)
            psi_FCI = np.dot(U.T.conj(), psi_FCI) * zeta
            psi_FCI = np.dot(U, psi_FCI) / LA.norm(psi_FCI)
            fide = fidelity(psi_FCI, U[:, m0]) if omega is None else LA.norm(psi_FCI[omega]) ** 2
            fout.write(f"{ib * db:.6f} {ea:.6f} {ev:.6f} {fide:.6f}\n")

        fout.write("FCI ITE Final Wavefunction:\n")
        print_state(psi_FCI, nbit, fout)


def ExpmbH(H_, psi_, db):
    """
    Compute the exponential of the Hamiltonian applied to a state.

    Parameters:
        H_ (np.ndarray): Hamiltonian operator.
        psi_ (np.ndarray): Wavefunction to evolve.
        db (float): Time step.

    Returns:
        np.ndarray: Evolved wavefunction.
        float: Normalization factor.
    """
    phi = psi_.copy()
    chi = psi_.copy()
    i = 0
    while LA.norm(chi) > 1e-6:
        chi = (-db / float(i + 1)) * Hpsi(H_, chi)
        phi += chi
        i += 1
    nu = LA.norm(phi)
    return phi / nu, nu


def Lanczos_kernel(Hmat_, chi_):
    """
    Lanczos kernel for computing the ground state.

    Parameters:
        Hmat_ (np.ndarray): Hamiltonian matrix.
        chi_ (np.ndarray): Trial wavefunction space.

    Returns:
        np.ndarray: Ground-state wavefunction.
    """
    chi_ = chi_[:, ::2]
    Sact = np.einsum("ic,id->cd", np.conj(chi_), chi_) + 1e-8 * np.eye(chi_.shape[1])
    Hact = np.einsum("ic,ij,jd->cd", np.conj(chi_), Hmat_, chi_)
    eps_, c_ = SciLA.eig(Hact, Sact)
    m0 = np.argmin(eps_)
    eta_ = np.einsum("c,ic->i", c_[:, m0], chi_) / LA.norm(chi_)
    return eta_


def ITE(H_, db, bmax, lanczos=False, psi0=None):
    """
    Perform Imaginary Time Evolution (ITE) with optional Lanczos enhancement.

    Parameters:
        H_ (np.ndarray): Hamiltonian operator.
        db (float): Time step for imaginary time evolution.
        bmax (float): Maximum imaginary time.
        lanczos (bool): Whether to use the Lanczos method. Defaults to False.
        psi0 (np.ndarray, optional): Initial wavefunction. Defaults to None.
    """
    N = H_[0][2].shape[1]
    nbit = int(np.log2(N))
    hdiag = np.array([Hii(H_, i) for i in range(N)])

    precond = lambda x, e, *args: x / (hdiag - e + 1e-4)
    hop = lambda c_: Hpsi(H_, c_)

    from pyscf.lib import davidson

    if psi0 is None:
        psi0 = np.zeros(N, dtype=complex)
        psi0[np.argmin(hdiag)] = 1.0

    epsm0, Um0 = davidson(hop, psi0, precond)

    with open("ITE.out", "w") as fout:
        fout.write(f"Ground-state energy: {epsm0:.6f}\n")
        fout.write("Ground-state wavefunction:\n")
        print_state(Um0, nbit, fout)

        psi_ITE = np.zeros(N, dtype=complex) if psi0 is None else psi0.copy()
        if psi0 is None:
            psi_ITE[np.argmin(hdiag)] = 1.0

        nbeta = int(bmax / db) + 1
        hvect_LANZ = np.zeros(nbeta + 1)
        svect_LANZ = np.zeros(nbeta + 1)
        if lanczos:
            space_LANZ = np.zeros((N, nbeta), dtype=complex)

        fout.write("ITE Progression:\n")
        for ib in range(nbeta):
            ea, ev = Hmoms(H_, psi_ITE)
            hvect_LANZ[ib] = ea
            fide = fidelity(psi_ITE, Um0)

            if lanczos:
                space_LANZ[:, ib] = psi_ITE
                psi_LANZ = Lanczos_kernel(Hmat(H_), space_LANZ[:, :ib + 1])
                ea_, ev_ = Hmoms(H_, psi_LANZ)
                fide_ = fidelity(psi_LANZ, Um0)
                fout.write(f"{ib * db:.6f} {ea:.6f} {ev:.6f} {fide:.6f} {ea_:.6f} {ev_:.6f} {fide_:.6f}\n")
            else:
                fout.write(f"{ib * db:.6f} {ea:.6f} {ev:.6f} {fide:.6f}\n")

            psi_ITE, dn = ExpmbH(H_, psi_ITE, db)
            svect_LANZ[ib + 1] = svect_LANZ[ib] + np.log(dn)

        fout.write("ITE Final Wavefunction:\n")
        print_state(psi_ITE, nbit, fout)
        dump_state(psi_ITE, nbit, "ite.psi")
        dump_lanz_vecs(hvect_LANZ[:nbeta], svect_LANZ[:nbeta], "qlanz.vecs")
