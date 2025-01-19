import numpy as np
from scipy import linalg as LA
from pyscf import lib


import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


from code_v4.qite import QITE_step
from code_v4.hamiltonian import Hmoms, Hpsi, Hmat, Heisenberg_SR


def metts_rt(H, nbits, nmetts, beta, db, outdir="./", tofile=True):
    """
    Perform METTS (Minimally Entangled Typical Thermal States) calculations using real-time evolution.

    Parameters:
        H (list): Hamiltonian terms.
        nbits (int): Number of bits (qubits/spins).
        nmetts (int): Number of METTS samples.
        beta (float): Inverse temperature (1/kT).
        db (float): Time step for imaginary time evolution.
        outdir (str): Directory for output files. Default is './'.
        tofile (bool): Whether to save results to a file. Default is True.

    Returns:
        tuple: Energy per site (E) and standard error (err).
    """
    Nbasis = 2**nbits
    nt = beta / (2 * db)
    if abs(nt - int(nt)) > 1e-9:
        raise ValueError("beta/2 cannot be divided by the time step!")
    nt = int(nt)

    # Initialize the wavefunction
    ci0 = np.zeros(Nbasis, dtype=np.complex128)
    ci0[np.random.randint(0, Nbasis)] = 1

    # Open output file if required
    if tofile:
        fout = open(outdir + f"nbit{nbits}_b{beta:.1f}.txt", "w")
        fout.write("# METTS#            E\n")

    Elst = []
    for k in range(nmetts):
        print(f"#### Constructing METTS # {k + 1} ...")

        # Imaginary time evolution
        for _ in range(nt):
            ci0 = QITE_step(H, ci0, db)

        # Measure energy
        ea, _ = Hmoms(H, ci0)
        Emetts = ea
        if tofile:
            fout.write(f"{k + 1}          {Emetts:.6f}\n")
        print(f"METTS # {k + 1}      E = {Emetts / nbits:.6f}")
        Elst.append(Emetts)

        # Collapse onto a product state
        basis = "X" if k % 2 == 0 else "Z"
        ci0 = collapse_metts(ci0, nbits, basis)

    if tofile:
        fout.close()

    Elst = np.asarray(Elst) / nbits  # Energy per site
    err = np.std(Elst) / np.sqrt(nmetts)
    E = np.average(Elst)
    return E, err


def collapse_metts(ci0, nbit, basis):
    """
    Collapse the wavefunction to a product state in a given basis.

    Parameters:
        ci0 (np.ndarray): Wavefunction.
        nbit (int): Number of bits (qubits/spins).
        basis (str): Basis for collapse ('X' or 'Z').

    Returns:
        np.ndarray: Collapsed wavefunction.
    """
    if basis == "X":
        Hadamard = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
        U = Hadamard.copy()
        for _ in range(nbit - 1):
            U = np.kron(U, Hadamard)
        ci0 = np.dot(U, ci0)

    ci0 /= np.linalg.norm(ci0)
    pval = (ci0 * ci0.conj()).real

    print("Probability distribution for CPS basis:")
    print(pval)

    cps = np.random.multinomial(1, pval).astype(np.complex128)
    print("The chosen CPS is number: ", np.where(cps != 0)[0][0])

    if basis == "X":
        cps = np.dot(U, cps)

    return cps


def finiteT_ED(H_, nbit, beta):
    """
    Calculate energy at finite temperature by diagonalizing the Hamiltonian.

    Parameters:
        H_ (list): Hamiltonian terms.
        nbit (int): Number of bits (qubits/spins).
        beta (float): Inverse temperature (1/kT).

    Returns:
        float: Energy per site.
    """
    H = Hmat(H_)
    ew, ev = np.linalg.eigh(H)
    Z = np.sum(np.exp(-beta * ew))
    E = np.sum(ew * np.exp(-beta * ew)) / Z
    return E / nbit


if __name__ == "__main__":
    # Hamiltonian parameters
    nbits = 2
    R = 1.5
    db = 0.1
    beta = 2.00
    nmetts = 200

    # Generate Hamiltonian
    H = Heisenberg_SR(nbits, R)

    # Compute ground-state energy via exact diagonalization
    Efci = finiteT_ED(H, nbits, beta)

    # Perform METTS calculations
    E, err = metts_rt(H, nbits, nmetts, beta, db)

    print(f"FCI result: {Efci:.6f}")
    print(f"METTS result: {E:.6f}   Standard deviation: {err:.6f}")
