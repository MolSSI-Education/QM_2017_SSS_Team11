import pytest
import scf_11
import numpy as np
import psi4

def test_scf():
    mole = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)
    mole.update_geometry()
    e_conv = 1.e-6
    d_conv = 1.e-6
    nel = 5
    bas = psi4.core.BasisSet.build(mole, target="aug-cc-PVDZ")
    mints = scf_11.get_mints(bas)
    E_total = scf_11.scf(mints, e_conv, d_conv, nel)
    assert np.allclose(psi4_energy, E_total)
