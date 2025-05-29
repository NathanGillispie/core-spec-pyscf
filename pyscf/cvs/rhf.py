import pyscf
from pyscf.tdscf.rhf import TDHF, TDA
import numpy

def core_valence(self, core_idx=None):
    '''This can be manually called to perform the CVS.
    Don't try this with something silly like fractional occupation numbers.'''
    if hasattr(self, 'core_idx'):
        core_idx = self.core_idx
    if core_idx is None:
        raise RuntimeError('Core orbitals not specified')

    self.check_sanity() # scf object exists and ran
    scf = self._scf

    if type(core_idx) is int:
        core_idx = [core_idx]

    core_idx = numpy.asarray(core_idx)
    scf.mol.nelec = (len(core_idx), len(core_idx))

    occ_idx = numpy.where(scf.mo_occ!=0)
    if not all(numpy.isin(core_idx, occ_idx)):
        print('Listed core orbitals aren\'t even occupied!')
    delete_idx = numpy.setxor1d(occ_idx, core_idx)

    scf.mo_occ = numpy.delete(scf.mo_occ, delete_idx, 0)
    scf.mo_coeff = numpy.delete(scf.mo_coeff, delete_idx, axis=1)
    scf.mo_energy = numpy.delete(scf.mo_energy, delete_idx, 0)

@pyscf.lib.with_doc(TDHF.kernel.__doc__)
def rpa_kernel(self, **kwargs):
    '''Monkey-patched TDHF/TDDFT kernel for CVS'''
    if 'core_idx' in kwargs.keys():
        self.core_idx = kwargs.pop('core_idx')
    if hasattr(self, 'core_idx'):
        self.core_valence()

    self._old_kernel(**kwargs)

@pyscf.lib.with_doc(TDA.kernel.__doc__)
def tda_kernel(self, **kwargs):
    '''Monkey-patched TDA kernel for CVS'''
    if 'core_idx' in kwargs.keys():
        self.core_idx = kwargs.pop('core_idx')
    if hasattr(self, 'core_idx'):
        self.core_valence()

    self._old_kernel(**kwargs)

TDHF.core_valence = core_valence
TDHF._old_kernel = TDHF.kernel
TDHF.kernel = rpa_kernel

TDA.core_valence = core_valence
TDA._old_kernel = TDA.kernel
TDA.kernel = tda_kernel
