import pyscf
from pyscf.tdscf.uhf import TDHF
import numpy

def core_valence(self, core_idx=None):
    '''
    This can be manually called to perform the CVS
    Don't try this with something silly like fractional occupation numbers.
    '''
    if hasattr(self, 'core_idx'):
        core_idx = self.core_idx
    if core_idx is None:
        raise RuntimeError('Core orbitals not specified')

    self.check_sanity() # scf object exists and ran
    scf = self._scf

    try:
        len(core_idx)
    except Exception as e:
        raise RuntimeError('core_idx must be in the form (idx_alpha, idx_beta)')
    assert len(core_idx) == 2

    if type(core_idx[0]) is int and type(core_idx[1]) is int:
        core_idx = ([core_idx[0]], [core_idx[1]])

    core_idx = numpy.asarray(core_idx)
    scf.mol.nelec = (len(core_idx[0]), len(core_idx[1]))

    occ_idx = (numpy.where(scf.mo_occ[0]!=0), numpy.where(scf.mo_occ[1]!=0))

    if not all(numpy.isin(core_idx[0], occ_idx[0])) or \
            not all(numpy.isin(core_idx[1], occ_idx[1])):
        print('Listed core orbitals aren\'t even occupied!')

    delete_a = numpy.setxor1d(occ_idx[0], core_idx[0])
    delete_b = numpy.setxor1d(occ_idx[1], core_idx[1])

    occa = np.delete(mf.mo_occ[0], delete_a, 0)
    occb = np.delete(mf.mo_occ[1], delete_b, 0)

    Ca = np.delete(mf.mo_coeff[0], delete_a, axis=1)
    Cb = np.delete(mf.mo_coeff[1], delete_b, axis=1)

    ea = np.delete(mf.mo_energy[0], delete_a, 0)
    eb = np.delete(mf.mo_energy[1], delete_b, 0)

    mf.mo_coeff = (Ca, Cb)
    mf.mo_occ = (occa, occb)
    mf.mo_energy = (ea, eb)


def kernel(self, **kwargs):
    '''Monkey-patched Kernel with CVS'''
    if 'core_idx' in kwargs.keys():
        self.core_idx = kwargs.pop('core_idx')
    if hasattr(self, 'core_idx'):
        self.core_valence()

    self._old_kernel(**kwargs)


TDHF.core_valence = core_valence
TDHF._old_kernel = TDHF.kernel
TDHF.kernel = kernel

