import pytest
import scf_11
import numpy as np
import psi4


def test_scf_basic():
    mole = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)
    mole.update_geometry()
    e_conv = 1.e-6
    d_conv = 1.e-6
    nel = 5
    JK_mode = False
    DIIS_mode = False
    dmp_start = 5
    dmp_value = 0.20
    bas = psi4.core.BasisSet.build(mole, target="aug-cc-PVDZ")
    mints = scf_11.get_mints(bas)
    E_total = scf_11.scf(mints, e_conv, d_conv, nel, JK_mode, DIIS_mode, dmp_start,
                         dmp_value, mole)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/aug-cc-PVDZ", molecule=mole)
    assert np.allclose(psi4_energy, E_total)

def test_scf_diis():
    mole = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)
    mole.update_geometry()
    e_conv = 1.e-6
    d_conv = 1.e-6
    nel = 5
    JK_mode = False
    DIIS_mode = True
    dmp_start = 5
    dmp_value = 0.20
    bas = psi4.core.BasisSet.build(mole, target="aug-cc-PVDZ")
    mints = scf_11.get_mints(bas)
    E_total = scf_11.scf(mints, e_conv, d_conv, nel, JK_mode, DIIS_mode, dmp_start,
                         dmp_value, mole)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/aug-cc-PVDZ", molecule=mole)
    assert np.allclose(psi4_energy, E_total)

def test_scf_jk():
    mole = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)
    mole.update_geometry()
    e_conv = 1.e-6
    d_conv = 1.e-6
    nel = 5
    JK_mode = True
    DIIS_mode = False
    dmp_start = 5
    dmp_value = 0.20
    bas = psi4.core.BasisSet.build(mole, target="aug-cc-PVDZ")
    mints = scf_11.get_mints(bas)
    E_total = scf_11.scf(mints, e_conv, d_conv, nel, JK_mode, DIIS_mode, dmp_start,
                         dmp_value, mole)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/aug-cc-PVDZ", molecule=mole)
    assert np.allclose(psi4_energy, E_total)

