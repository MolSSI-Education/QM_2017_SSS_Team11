

import psi4
import numpy as np

class Molecule:
    '''
    A molecule class that is used to store basic parameters and
    atomic orbitals/density matrices
    '''

    def __init__(self, mol=None, bas=None, nel=0):
        '''
        Initialize molecule with geometry mol, basis
        set bas and number of occupied orbitals nel
        '''
        self.mol = mol
        self.bas = bas
        self.nel = nel
        self.C = None
        self.D = None
        self.g = None
        self.F = None
        self.eps = None
        self.ao_eri_computed = False
        self.ao_eri = None

    def set_geometry(self, geom_str):
        '''
        set the geometry of the molecule by providing
        a psi4 style string
        '''
        self.mol = psi4.geometry(geom_str)
        self.mol.update_geometry()

    def set_basis(bas_str):
        '''
        set the basis set that is used in any further calculations
        '''
        if(self.mol is None):
            raise Exception('Error: Geometry not defined')
        else:
            self.bas = psi4.core.BasisSet.build(self.mol, target=bas_str)

    def get_mints(self):
        '''
        Built a MintsHelper - helps get integrals from psi4
        '''
        mints = psi4.core.MintsHelper(self.bas)

        nbf = mints.nbf()

        if (nbf > 200):
            raise Exception("More than 200 basis functions!")

        return mints

    def get_ao_eri(self):
        '''
        Returns two electron integrals. If they don't exist
        use mints to calculate and store them.
        '''
        if self.ao_eri_computed == False:
            self.ao_eri = np.array(self.get_mints().ao_eri())
        return self.ao_eri
