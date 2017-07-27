# coding: utf-8

import numpy as np
import psi4
try:
    from . import params
    from . import diis
    from . import jk
except SystemError:
    import params
    import diis
    import jk

np.set_printoptions(suppress=True, precision=4)


def get_mints(bas):
    '''
    Built a MintsHelper - helps get integrals from psi4
    '''

    mints = psi4.core.MintsHelper(bas)

    nbf = mints.nbf()

    if (nbf > 100):
        raise Exception("More than 100 basis functions!")

    return mints


def diag(F, A):
    '''
    Function to diag. a Fock mat. F with orthog. mat. A
    '''

    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


def make_JK(g, Cocc, JK_mode):
    '''
    Function to make Coulomb matrix J and Exchange matrix K from two-electron integrals g
    and density D, with flag JK_mode
    '''
    if JK_mode:
        J, K = jk.make_JK_adv(g, Cocc)
    if not JK_mode:
        D = Cocc @ Cocc.T
        J = np.einsum("pqrs,rs->pq", g, D)
        K = np.einsum("prqs,rs->pq", g, D)
    return J, K


def make_K(g, D, JK_mode):
    '''
    Function to make Exchange matrix J from two-electron integrals g
    and density D, with flag JK_mode
    '''

    if not JK_mode:
        K = np.einsum("prqs,rs->pq", g, D)
    return K


def damp(F_old, F_new, damp_start, damp_value, i):
    '''
    Function for handling convergence damping
    '''

    if i >= damp_start:
        F = damp_value * F_old + (1.0 - damp_value) * F_new
    else:
        F = F_new
    return F


def scf(mints, e_conv, d_conv, nel, JK_mode, DIIS_mode, damp_start,
        damp_value, mol):
    '''
    Main SCF function, returns HF Energy
    '''

    # Constructing kinetic and potential energy arrays
    V = np.array(mints.ao_potential())
    T = np.array(mints.ao_kinetic())

    # Constructing core Hamiltonian
    H = T + V

    # Constructing overlap and electron repulsion integral arrays
    S = np.array(mints.ao_overlap())
    g = np.array(mints.ao_eri())

    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    A = np.array(A)

    # Constructing initial density matrix
    eps, C = diag(H, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T

    # Starting SCF loop
    E_old = 0.0
    F_old = None
    F_list = []
    grad_list = []

    for iteration in range(30):
        # Form J and K
        J, K = make_JK(g, Cocc, JK_mode)
#        K = make_K(g, Cocc, JK_mode)

        # Form new Fock matrix
        F_new = H + 2.0 * J - K

        # Check if DIIS extrapolation is requested
        if DIIS_mode:
            F_list.append(F_new)
            F = F_new
        else:
            F = damp(F_old, F_new, damp_start, damp_value, iteration)
            F_old = F_new

        # Build the AO gradient
        grad = A.T @ (F @ D @ S - S @ D @ F) @ A

        # Keep track of gradient root-mean-squared
        grad_rms = np.mean(grad ** 2) ** 0.5

        # Build the energy
        E_electric = np.sum((F + H) * D)
        E_total = E_electric + mol.nuclear_repulsion_energy()

        E_diff = E_total - E_old
        E_old = E_total

        print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
              (iteration, E_total, E_diff, grad_rms))

        # Break if e_conv and d_conv are met
        if (E_diff < e_conv) and (grad_rms < d_conv):
            print("SCF has finished!")
            break

        # Append gradient list if DIIS extrapolation is requested
        if DIIS_mode:
            grad_list.append(grad)
            if iteration > 2:
                F = diis.diis(F_list, grad_list)

        # Build final density matrix
        eps, C = diag(F, A)
        Cocc = C[:, :nel]
        D = Cocc @ Cocc.T
        if (iteration == 29):
            print("SCF steps have reached max.")
    return E_total

if __name__ == "__main__":
    mints = get_mints(params.bas)

    scf(mints, params.e_conv, params.d_conv, params.nel, params.JK_mode,
        params.DIIS_mode,
        params.damp_start, params.damp_value, params.mol)
