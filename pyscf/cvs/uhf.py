import pyscf
from pyscf.tdscf.uhf import TDHF, TDA
import numpy

def core_valence(self, core_idx=None):
    '''This can be manually called to perform the CVS
    Don't try this with something silly like fractional occupation numbers.'''
    if hasattr(self, 'core_idx'):
        core_idx = self.core_idx
    if core_idx is None:
        # Happens only when a user calls this function
        raise RuntimeWarning('Core orbitals not specified. Use the core_idx attribute.')
        return

    self.check_sanity() # scf object exists and ran
    scf = self._scf

    if len(core_idx) != 2:
        raise ValueError('core_idx must be in the form (idx_alpha, idx_beta)')

    if type(core_idx[0]) is int and type(core_idx[1]) is int:
        core_idx = ([core_idx[0]], [core_idx[1]])

    core_idx = numpy.asarray(core_idx)

    occ_idx = (numpy.where(scf.mo_occ[0]!=0), numpy.where(scf.mo_occ[1]!=0))

    if not all(numpy.isin(core_idx[0], occ_idx[0])) or \
            not all(numpy.isin(core_idx[1], occ_idx[1])):
        print('Listed core orbitals aren\'t even occupied!')

    # We want to have the same number of alpha and beta orbitals, nelec_a >= nelec_b
    delete_b = numpy.setxor1d(occ_idx[1], core_idx[1])

    occa = numpy.delete(scf.mo_occ[0], delete_b, 0)
    occb = numpy.delete(scf.mo_occ[1], delete_b, 0)

    Ca = numpy.delete(scf.mo_coeff[0], delete_b, axis=1)
    Cb = numpy.delete(scf.mo_coeff[1], delete_b, axis=1)

    ea = numpy.delete(scf.mo_energy[0], delete_b, 0)
    eb = numpy.delete(scf.mo_energy[1], delete_b, 0)

    scf.mol.nelec = ((occa!=0).sum(), (occb!=0).sum())
    scf.mo_coeff = (Ca, Cb)
    scf.mo_occ = (occa, occb)
    scf.mo_energy = (ea, eb)

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

