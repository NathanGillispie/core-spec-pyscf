import pyscf
from pyscf.tdscf.rhf import TDHF, TDA
from pyscf.tdscf.rks import CasidaTDDFT
import numpy
from scipy.linalg import sqrtm

from pyscf.lib import logger
from .no_fxc import get_ab_no_fxc_rhf

def core_valence(self, core_idx=None):
    '''This can be manually called to perform the CVS.'''
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

def direct_diag_tda_kernel(self, x0=None, nstates=None):
    '''TDA diagonalization solver'''
    cpu0 = (logger.process_clock(), logger.perf_counter())
    self.check_sanity()
    self.dump_flags()
    if nstates is None:
        nstates = self.nstates
    else:
        self.nstates = nstates
    mol = self.mol

    log = logger.Logger(self.stdout, self.verbose)

    a, _ = self.get_ab()
    nocc = a.shape[0]
    nvir = a.shape[1]
    a = a.reshape(nocc*nvir, nocc*nvir)

    e, x1 = numpy.linalg.eigh(a)
    self.e = numpy.extract(e > self.positive_eig_threshold, e)
    self.xy = [(xi.reshape(nocc,nvir)*numpy.sqrt(.5),0) for xi in x1]
    self.converged = [True]

    if self.chkfile:
        pyscf.lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
        pyscf.lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

    log.timer('TDA', *cpu0)
    self._finalize()
    return self.e, self.xy

def direct_diag_rpa_kernel(self, x0=None, nstates=None):
    '''TDHF/TDDFT direct-diagonalization solver'''
    log = logger.new_logger(self)
    cpu0 = (logger.process_clock(), logger.perf_counter())
    self.check_sanity()
    self.dump_flags()
    if nstates is None:
        nstates = self.nstates
    else:
        self.nstates = nstates
    mol = self.mol

    A, B = self.get_ab()
    assert A.dtype == 'float64'
    nocc = A.shape[0]
    nvir = A.shape[1]
    A = A.reshape(nocc*nvir, nocc*nvir)
    B = B.reshape(nocc*nvir, nocc*nvir)

    sqamb = sqrtm(A-B)
    if sqamb.dtype != 'float64':
        log.warn("A-B is not positive semi-definite! Results may not be accurate. Try another basis?")
        sqamb = numpy.asarray(sqamb, dtype='float64')
    C = sqamb @ (A + B) @ sqamb

    e_squared, Z = numpy.linalg.eigh(C)
    e = (e_squared)**.5

    xmy = numpy.linalg.inv(sqamb) @ Z
    xpy = sqamb @ Z # @ numpy.diag(1/e) ? that's what I worked out

    X = .5 * (xpy + xmy)
    Y = .5 * (xpy - xmy)
    x1 = numpy.zeros((X.shape[0], X.shape[1]*2))
    x1[:,:nocc*nvir] += X
    x1[:,nocc*nvir:] += Y

    self.e = numpy.extract(e > self.positive_eig_threshold, e)

    def norm_xy(z):
        x, y = z.reshape(2, -1)
        norm = pyscf.lib.norm(x)**2 - pyscf.lib.norm(y)**2
        if norm < 0:
            log.warn('TDDFT amplitudes |X| smaller than |Y|')
        norm = abs(.5/norm) ** .5 # normalize to 0.5 for alpha spin
        return x.reshape(nocc,nvir)*norm, y.reshape(nocc,nvir)*norm
    self.xy = [norm_xy(z) for z in x1]
    self.converged = [True]

    if self.chkfile:
        pyscf.lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
        pyscf.lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

    log.timer('TDA', *cpu0)
    self._finalize()
    return self.e, self.xy

@pyscf.lib.with_doc(TDHF.kernel.__doc__)
def rpa_kernel(self, **kwargs):
    '''Monkey-patched TDHF/TDDFT kernel for CVS'''
    if 'core_idx' in kwargs.keys():
        self.core_idx = kwargs.pop('core_idx')
    if hasattr(self, 'core_idx'):
        self.core_valence()

    if 'no_fxc' in kwargs.keys():
        self.no_fxc = kwargs.pop('no_fxc')
    if hasattr(self, 'no_fxc'):
        if self.no_fxc:
            self.get_ab = get_ab_no_fxc_rhf

    if 'direct_diag' in kwargs.keys():
        self.direct_diag = kwargs.pop('direct_diag')
    if hasattr(self, 'direct_diag'):
        if self.direct_diag:
            return direct_diag_rpa_kernel(self, **kwargs)
    return self._old_kernel(**kwargs)

@pyscf.lib.with_doc(TDA.kernel.__doc__)
def tda_kernel(self, **kwargs):
    '''Monkey-patched TDA kernel for CVS'''
    if 'core_idx' in kwargs.keys():
        self.core_idx = kwargs.pop('core_idx')
    if hasattr(self, 'core_idx'):
        self.core_valence()

    if 'no_fxc' in kwargs.keys():
        self.no_fxc = kwargs.pop('no_fxc')
    if hasattr(self, 'no_fxc'):
        if self.no_fxc:
            self.get_ab = get_ab_no_fxc_rhf

    if 'direct_diag' in kwargs.keys():
        self.direct_diag = kwargs.pop('direct_diag')
    if hasattr(self, 'direct_diag'):
        if self.direct_diag:
            return direct_diag_tda_kernel(self, **kwargs)
    return self._old_kernel(**kwargs)

TDHF.core_valence = core_valence
TDHF._old_kernel = TDHF.kernel
TDHF.kernel = rpa_kernel

CasidaTDDFT.core_valence = core_valence
CasidaTDDFT._old_kernel = CasidaTDDFT.kernel
CasidaTDDFT.kernel = rpa_kernel

TDA.core_valence = core_valence
TDA._old_kernel = TDA.kernel
TDA.kernel = tda_kernel


