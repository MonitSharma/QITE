import numpy as np
from numpy import linalg as LA
from scipy import linalg as SciLA
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize

import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from pauli import pauli_action
from binary_functions import Bas2Int, Int2Bas
from tools import print_state, fidelity, dump_state, dump_lanczos_vectors
from hamiltonian import Hmat, Hmoms

# -------------------------------------------------
# Term-by-Term Imaginary Time Evolution (ITE)

def H_alpha_psi(H_, psi_, alpha_):
    """
    Apply a single Hamiltonian term to the wavefunction.
    """
    A, h, imp, gmp = H_[alpha_]
    phi = np.zeros_like(psi_, dtype=complex)
    for m in np.where(np.abs(h) > 1e-8)[0]:
        phi += h[m] * gmp[m, imp[m, :]] * psi_[imp[m, :]]
    return phi


def ExpmbH_alpha(H_, psi_, alpha_, db):
    """
    Perform exponential evolution for a single Hamiltonian term.
    """
    phi = psi_.copy()
    chi = psi_.copy()
    i = 0
    while LA.norm(chi) > 1e-8:
        chi = (-db / float(i + 1)) * H_alpha_psi(H_, chi, alpha_)
        phi += chi
        i += 1
    nu = LA.norm(phi)
    return phi / nu, nu

# -------------------------------------------------
# Unitary Evolution

def xP_psi(x_, psi_, imp_, gmp_):
    """
    Apply the action of Pauli operators weighted by coefficients.
    """
    phi = np.zeros_like(psi_, dtype=complex)
    for m in np.where(np.abs(x_) > 1e-8)[0]:
        phi += x_[m] * gmp_[m, imp_[m, :]] * psi_[imp_[m, :]]
    return phi


def Exp_ixP(x_, psi_, imp_, gmp_):
    """
    Perform unitary evolution with exponential of Pauli operators.
    """
    phi = psi_.copy()
    chi = psi_.copy()
    i = 0
    while LA.norm(chi) > 1e-8:
        chi = (1j / float(i + 1)) * xP_psi(x_, chi, imp_, gmp_)
        phi += chi
        i += 1
    return phi / LA.norm(phi)

# -------------------------------------------------
# QITE Step

def QITE_step(H_, psi_, db, xv=None, check=False):
    """
    Perform a single step of Quantum Imaginary Time Evolution (QITE).
    """
    nalpha = len(H_)
    dn_ = 1.0
    if xv is None:
        xv = [np.zeros(H_[alpha][2].shape[0]) for alpha in range(nalpha)]

    for alpha in range(nalpha):
        delta_alpha, dn_alpha = ExpmbH_alpha(H_, psi_, alpha, db)
        delta_alpha -= psi_
        dn_ *= dn_alpha

        # Pauli action
        A, h, imp, gmp = H_[alpha]
        nact = imp.shape[0]
        Pmu_psi = np.array([gmp[m, imp[m, :]] * psi_[imp[m, :]] for m in range(nact)])

        # Set up the sparse linear system
        Amat = coo_matrix((
            2 * np.real(np.einsum("a,na->n", np.conj(Pmu_psi), Pmu_psi)).flatten(),
            np.indices(Pmu_psi.shape)
        ))
        bvec = -2.0 * np.imag(np.dot(Pmu_psi, np.conj(delta_alpha)))

        if check:
            x = lsqr(Amat, bvec)[0]
        else:
            zct = Amat.T @ bvec

            def cost_fun(v):
                return LA.norm(Amat @ v - bvec)**2

            def J_cost_fun(v):
                return 2 * (Amat.T @ (Amat @ v) - zct)

            x = minimize(cost_fun, x0=xv[alpha], method="Newton-CG", jac=J_cost_fun).x

        xv[alpha] = x
        psi_ = Exp_ixP(x, psi_, imp, gmp)

    return psi_, dn_, xv

# -------------------------------------------------
# Lanczos QITE

def Lanczos_QITE(hv, sv, db):
    """
    Perform Lanczos Quantum Imaginary Time Evolution (QITE).
    """
    nv = len(range(0, len(hv), 2))
    hm = np.zeros((nv, nv), dtype=complex)
    sm = np.zeros((nv, nv), dtype=complex)

    for jr in range(0, len(hv), 2):
        for js in range(0, len(hv), 2):
            jk = (jr + js) // 2
            sm[jr // 2, js // 2] = np.exp(2 * sv[jk] - sv[jr] - sv[js])
            hm[jr // 2, js // 2] = hv[jk] * sm[jr // 2, js // 2]

    # Regularization
    for jk in range(nv):
        sm[jk, jk] *= 1.0 + 2 * db
        hm[jk, jk] *= 1.0 + 2 * db

    eps, _ = SciLA.eigh(hm, sm)
    return np.min(np.real(eps))

# -------------------------------------------------
# QITE Main Routine

def QITE(H_, db, bmax, lanczos=False, psi0=None, omega=None, ncheck=1):
    """
    Perform Quantum Imaginary Time Evolution (QITE).
    """
    Hm = Hmat(H_)
    N = Hm.shape[0]
    nbit = int(np.log2(N))
    eps, U = SciLA.eigh(Hm)
    m0 = np.argmin(eps)

    with open("QITE.out", "w") as fout:
        fout.write(f"Ground-state energy: {eps[m0]:.6f}\n")
        fout.write("Ground-state wavefunction:\n")
        print_state(U[:, m0], nbit, fout)

        psi_QITE = psi0.copy()
        nbeta = int(bmax / db) + 1
        hvect_LANZ = np.zeros(nbeta + 1)
        svect_LANZ = np.zeros(nbeta + 1)
        xv = None

        fout.write("QITE Progression:\n")
        for ib in range(nbeta):
            ea, ev = Hmoms(H_, psi_QITE)
            hvect_LANZ[ib] = ea

            if omega is None:
                fide = fidelity(psi_QITE, U[:, m0])
            else:
                fide = LA.norm(psi_QITE[omega])**2

            if lanczos:
                ea_ = Lanczos_QITE(hvect_LANZ[:ib + 1], svect_LANZ[:ib + 1], db)
                fout.write(f"{ib * db:.6f} {ea:.6f} {ev:.6f} {fide:.6f} {ea_:.6f}\n")
            else:
                fout.write(f"{ib * db:.6f} {ea:.6f} {ev:.6f} {fide:.6f}\n")

            psi_QITE, dnorm, xv = QITE_step(H_, psi_QITE, db, xv, check=(ib % ncheck == 0))
            svect_LANZ[ib + 1] = svect_LANZ[ib] + np.log(dnorm)

        fout.write("Final wavefunction:\n")
        print_state(psi_QITE, nbit, fout)
        dump_state(psi_QITE, nbit, "qite.psi")
        dump_lanz_vecs(hvect_LANZ[:nbeta], svect_LANZ[:nbeta], "qlanz.vecs")
