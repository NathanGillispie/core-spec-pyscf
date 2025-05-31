import pyscf
from pyscf.tdscf.uhf import TDHF, TDA, REAL_EIG_THRESHOLD
from pyscf.tdscf.uks import CasidaTDDFT
import numpy
from scipy.linalg import sqrtm

from pyscf.lib import logger
from .no_fxc import get_ab_no_fxc_uhf

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

def direct_diag_tda_kernel(self, x0=None, nstates=None):
    '''TDA diagonalization solver'''
    cpu0 = (logger.process_clock(), logger.perf_counter())
    self.check_sanity()
    self.dump_flags()
    if nstates is None:
        nstates = self.nstates
    else:
        self.nstates = nstates
    log = logger.Logger(self.stdout, self.verbose)

    (aa, ab, bb), _ = self.get_ab()
    assert ab.dtype == 'float64'

    nocca, nvira, _, _ = aa.shape
    noccb, nvirb, _, _ = bb.shape

    ova = nocca * nvira
    ovb = noccb * nvirb
    A = numpy.zeros((ova+ovb, ova+ovb))

    aa = aa.reshape((ova, ova))
    ba = ab.transpose((2,3,0,1)).reshape((ovb, ova))
    ab = ab.reshape((ova, ovb))
    bb = bb.reshape((ovb, ovb))

    A[:ova, :ova] += aa
    A[:ova, ova:] += ab
    A[ova:, :ova] += ba
    A[ova:, ova:] += bb

    e, x1 = numpy.linalg.eigh(A)
    # e, x1 = numpy.linalg.eig(A)
    # if numpy.amax(x1.imag) > REAL_EIG_THRESHOLD:
    #     log.warn(self, 'Imaginary component too large, bad basis?')
    # else:
    #     x1 = numpy.asarray(x1, dtype='float64')

    self.xy = [((xi[:ova].reshape(nocca,nvira),  # X_alpha
                 xi[ova:].reshape(noccb,nvirb)), # X_beta
                (0, 0))  # (Y_alpha, Y_beta)
               for xi in x1]

    self.converged = [True]
    self.e = numpy.extract(e > self.positive_eig_threshold, e)

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

    (Aaa, Aab, Abb), (Baa, Bab, Bbb) = self.get_ab()
    assert Aab.dtype == 'float64'
    assert Bab.dtype == 'float64'

    nocca, nvira, _, _ = Aaa.shape
    noccb, nvirb, _, _ = Abb.shape

    ova = nocca * nvira
    ovb = ova

    Aaa = Aaa.reshape((ova, ova))
    Aba = Aab.transpose((2,3,0,1)).reshape((ovb, ova))
    Aab = Aab.reshape((ova, ovb))
    Abb = Abb.reshape((ovb, ovb))
    Baa = Baa.reshape((ova, ova))
    Bba = Bab.transpose((2,3,0,1)).reshape((ovb, ova))
    Bab = Bab.reshape((ova, ovb))
    Bbb = Bbb.reshape((ovb, ovb))

    A = numpy.zeros((ova+ovb, ova+ovb))
    B = numpy.zeros_like(A)
    A[:ova, :ova] += Aaa
    A[:ova, ova:] += Aab
    A[ova:, :ova] += Aba
    A[ova:, ova:] += Abb

    B[:ova, :ova] += Baa
    B[:ova, ova:] += Bab
    B[ova:, :ova] += Bba
    B[ova:, ova:] += Bbb

    sqamb = sqrtm(A-B)
    if sqamb.dtype != 'float64':
        log.warn("A-B is not positive semi-definite! Results may not be accurate. Try another basis?")
        sqamb = numpy.asarray(sqamb, dtype='float64')
    C = sqamb @ (A + B) @ sqamb
    e_squared, Z = numpy.linalg.eigh(C)
    e = (e_squared)**.5

    xmy = numpy.linalg.inv(sqamb) @ Z
    xpy = sqamb @ Z

    X = .5 * (xpy + xmy)
    Y = .5 * (xpy - xmy)
    x1 = numpy.zeros((X.shape[0], X.shape[1]*2))
    x1[:,:ova*2] += X
    x1[:,ova*2:] += Y

    xy = []
    for i, z in enumerate(x1):
        x, y = z.reshape(2, -1)
        norm = pyscf.lib.norm(x)**2 - pyscf.lib.norm(y)**2
        if norm < 0:
            log.warn('TDDFT amplitudes |X| smaller than |Y|')
        norm = abs(norm)**-.5
        xy.append(((x[:nocca*nvira].reshape(nocca,nvira) * norm,  # X_alpha
                    x[nocca*nvira:].reshape(noccb,nvirb) * norm), # X_beta
                   (y[:nocca*nvira].reshape(nocca,nvira) * norm,  # Y_alpha
                    y[nocca*nvira:].reshape(noccb,nvirb) * norm)))# Y_beta
    self.xy = xy
    self.converged = [True]

    if self.chkfile:
        pyscf.lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
        pyscf.lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

    log.timer('TDHF/TDDFT', *cpu0)
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
            self.get_ab = get_ab_no_fxc_uhf

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
            self.get_ab = get_ab_no_fxc_uhf

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

