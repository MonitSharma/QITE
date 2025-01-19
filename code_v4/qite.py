import numpy as np
from numpy import linalg as LA
from scipy import linalg as SciLA
from scipy.optimize import minimize
from pauli import pauli_action, sigma_matrices
from binary_functions import Bas2Int, Int2Bas
from tools import print_state, fidelity, dump_state, dump_lanczos_vectors
from hamiltonian import Hmat, Hmoms, Hpsi, Hii

# -------------------------------------------------
# Term-by-Term Imaginary Time Evolution

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

        A, h, imp, gmp = H_[alpha]
        nact = imp.shape[0]

        # Compute P_mu psi
        Pmu_psi = np.zeros_like(imp, dtype=complex)
        for m in range(nact):
            Pmu_psi[m, :] = gmp[m, imp[m, :]] * psi_[imp[m, :]]

        # Set up linear system
        Amat = 2.0 * np.real(np.dot(Pmu_psi, Pmu_psi.T.conj()))
        bvec = -2.0 * np.imag(np.dot(Pmu_psi, delta_alpha.conj()))

        # Solve the linear system
        if check:
            x = SciLA.lstsq(Amat, bvec)[0]
        else:
            def cost_fun(vct):
                return LA.norm(np.dot(Amat, vct) - bvec)**2

            x = minimize(cost_fun, x0=xv[alpha], method='CG', tol=1e-8).x
        xv[alpha] = x.copy()

        # Update the wavefunction
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

    # Rarefied sampling
    idx = list(range(nv))
    sm = sm[np.ix_(idx, idx)]
    hm = hm[np.ix_(idx, idx)]

    # Regularization
    for jk in range(nv):
        sm[jk, jk] *= 1.0 + 2 * db
        hm[jk, jk] *= 1.0 + 2 * db

    eps, _ = SciLA.eigh(hm, sm)
    return np.min(np.real(eps))

# -------------------------------------------------
# QITE Main Routine

def QITE(H_, db, bmax, lanczos=False, psi0=None, omega=None, ncheck=1, davidson=True):
    """
    Perform Quantum Imaginary Time Evolution (QITE).
    """
    if davidson:
        N = H_[0][2].shape[1]
        hdiag = np.array([Hii(H_, i) for i in range(N)])
        precond = lambda x, e, *args: x / (hdiag - e + 1e-4)

        def hop(c_):
            return Hpsi(H_, c_)

        if psi0 is None:
            psi0 = np.zeros(N, dtype=complex)
            psi0[np.argmin(hdiag)] = 1.0

        from pyscf.lib import davidson
        epsm0, Um0 = davidson(hop, psi0, precond)
    else:
        Hm = Hmat(H_)
        eps, U = SciLA.eigh(Hm)
        m0 = np.argmin(eps)
        epsm0 = eps[m0]
        Um0 = U[:, m0]

    with open("QITE.out", "w") as fout:
        fout.write(f"Ground-state energy: {epsm0:.6f}\n")
        fout.write("Ground-state wavefunction:\n")
        print_state(Um0, int(np.log2(H_[0][2].shape[1])), fout)

        psi_QITE = psi0[:]
        nbeta = int(bmax / db) + 1
        hvect_LANZ = np.zeros(nbeta + 1)
        svect_LANZ = np.zeros(nbeta + 1)
        xv = None

        fout.write("QITE Progression:\n")
        for ib in range(nbeta):
            ea, ev = Hmoms(H_, psi_QITE)
            hvect_LANZ[ib] = ea
            fide = fidelity(psi_QITE, Um0) if omega is None else LA.norm(psi_QITE[omega])**2

            if lanczos:
                ea_ = Lanczos_QITE(hvect_LANZ[:ib + 1], svect_LANZ[:ib + 1], db)
                fout.write(f"{ib * db:.6f} {ea:.6f} {ev:.6f} {fide:.6f} {ea_:.6f}\n")
            else:
                fout.write(f"{ib * db:.6f} {ea:.6f} {ev:.6f} {fide:.6f}\n")

            if ncheck > 0 and ib % ncheck == 0:
                psi_QITE, dnorm, xv = QITE_step(H_, psi_QITE, db, xv, check=True)
            else:
                psi_QITE, dnorm, xv = QITE_step(H_, psi_QITE, db, xv, check=False)
            svect_LANZ[ib + 1] = svect_LANZ[ib] + np.log(dnorm)

        fout.write("Final wavefunction:\n")
        print_state(psi_QITE, int(np.log2(H_[0][2].shape[1])), fout)
        dump_state(psi_QITE, int(np.log2(H_[0][2].shape[1])), "qite.psi")
        dump_lanczos_vectors(hvect_LANZ[:nbeta], svect_LANZ[:nbeta], "qlanz.vecs")
