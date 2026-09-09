import pyscf
from pyscf.tdscf.uhf import TDHF, TDA
from pyscf.tdscf.uks import CasidaTDDFT
import numpy
from scipy.linalg import sqrtm

from pyscf.lib import logger
from .no_fxc import get_ab_no_fxc_uhf
from ._utils import core_valence_unrestricted, patch_td_class, prepare_kernel

core_valence = core_valence_unrestricted


def _get_ab(tdobj, no_fxc=False):
    frozen = getattr(tdobj, 'frozen', None)
    mf = tdobj._scf
    if no_fxc:
        return get_ab_no_fxc_uhf(mf, frozen=frozen)
    return tdobj.get_ab(mf=mf, frozen=frozen)


def direct_diag_tda_kernel(self, x0=None, nstates=None, no_fxc=False):
    '''TDA diagonalization solver'''
    log = logger.new_logger(self)
    cpu0 = (logger.process_clock(), logger.perf_counter())
    self.check_sanity()
    self.dump_flags()
    if nstates is None:
        nstates = self.nstates
    else:
        self.nstates = nstates

    (aa, ab, bb), _ = _get_ab(self, no_fxc=no_fxc)
    assert ab.dtype == numpy.float64

    nocca, nvira, _, _ = aa.shape
    noccb, nvirb, _, _ = bb.shape

    ova = nocca * nvira
    ovb = noccb * nvirb
    A = numpy.zeros((ova + ovb, ova + ovb))

    aa = aa.reshape((ova, ova))
    ba = ab.transpose((2, 3, 0, 1)).reshape((ovb, ova))
    ab = ab.reshape((ova, ovb))
    bb = bb.reshape((ovb, ovb))

    A[:ova, :ova] += aa
    A[:ova, ova:] += ab
    A[ova:, :ova] += ba
    A[ova:, ova:] += bb

    e, x1 = numpy.linalg.eigh(A)

    keep_idx = numpy.where(e > self.positive_eig_threshold)[0]
    e = e[keep_idx]
    x1 = x1[:, keep_idx]

    self.e = e[:nstates]
    x1 = x1[:, :nstates]

    self.xy = [
        (
            (
                xi[:ova].reshape(nocca, nvira),  # X_alpha
                xi[ova:].reshape(noccb, nvirb)),  # X_beta
            (0, 0))  # (Y_alpha, Y_beta)
        for xi in x1.T
    ]
    self.converged = [True]

    if self.chkfile:
        pyscf.lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
        pyscf.lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

    log.timer('TDA', *cpu0)
    self._finalize()
    return self.e, self.xy


def direct_diag_rpa_kernel(self, x0=None, nstates=None, no_fxc=False):
    '''TDHF/TDDFT direct-diagonalization solver'''
    log = logger.new_logger(self)
    cpu0 = (logger.process_clock(), logger.perf_counter())
    self.check_sanity()
    self.dump_flags()
    if nstates is None:
        nstates = self.nstates
    else:
        self.nstates = nstates

    (Aaa, Aab, Abb), (Baa, Bab, Bbb) = _get_ab(self, no_fxc=no_fxc)
    assert Aab.dtype == numpy.float64
    assert Bab.dtype == numpy.float64

    nocca, nvira, _, _ = Aaa.shape
    noccb, nvirb, _, _ = Abb.shape

    ova = nocca * nvira
    ovb = noccb * nvirb

    Aaa = Aaa.reshape((ova, ova))
    Aba = Aab.transpose((2, 3, 0, 1)).reshape((ovb, ova))
    Aab = Aab.reshape((ova, ovb))
    Abb = Abb.reshape((ovb, ovb))
    Baa = Baa.reshape((ova, ova))
    Bba = Bab.transpose((2, 3, 0, 1)).reshape((ovb, ova))
    Bab = Bab.reshape((ova, ovb))
    Bbb = Bbb.reshape((ovb, ovb))

    A = numpy.zeros((ova + ovb, ova + ovb))
    B = numpy.zeros_like(A)
    A[:ova, :ova] += Aaa
    A[:ova, ova:] += Aab
    A[ova:, :ova] += Aba
    A[ova:, ova:] += Abb

    B[:ova, :ova] += Baa
    B[:ova, ova:] += Bab
    B[ova:, :ova] += Bba
    B[ova:, ova:] += Bbb

    sqamb = sqrtm(A - B)
    if sqamb.dtype != numpy.float64:
        log.warn(
            "A-B is not positive semi-definite! Results may not be accurate. Try another basis?"
        )
        sqamb = numpy.asarray(sqamb.real, dtype=numpy.float64)
    C = sqamb @ (A + B) @ sqamb
    e_squared, Z = numpy.linalg.eigh(C)
    e = numpy.sqrt(numpy.clip(numpy.real(e_squared), 0, None))
    inv_e = numpy.divide(
        1.0, e, out=numpy.zeros_like(e), where=e > self.positive_eig_threshold)

    xmy = numpy.linalg.inv(sqamb) @ Z
    xpy = sqamb @ Z @ numpy.diag(inv_e)

    X = .5 * (xpy + xmy)
    Y = .5 * (xpy - xmy)
    x1 = numpy.zeros(((ova + ovb) * 2, ovb + ova))
    x1[:ovb + ova] += X
    x1[ovb + ova:] += Y

    keep_idx = numpy.where(e > self.positive_eig_threshold)[0]
    self.e = (e[keep_idx])[:nstates]

    xy = []
    for i, z in enumerate(x1.T):
        if i not in keep_idx:
            continue
        x, y = z.reshape(2, -1)
        norm = pyscf.lib.norm(x)**2 - pyscf.lib.norm(y)**2
        if norm < 0:
            log.warn('TDDFT amplitudes |X| smaller than |Y|')
        norm = abs(norm)**-.5
        xy.append((
            (
                x[:nocca * nvira].reshape(nocca, nvira) * norm,  # X_alpha
                x[nocca * nvira:].reshape(noccb, nvirb) * norm),  # X_beta
            (
                y[:nocca * nvira].reshape(nocca, nvira) * norm,  # Y_alpha
                y[nocca * nvira:].reshape(noccb, nvirb) * norm)))  # Y_beta
    self.xy = xy[:nstates]
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
    no_fxc, direct_diag = prepare_kernel(self, kwargs, core_valence)
    if direct_diag:
        return direct_diag_rpa_kernel(self, no_fxc=no_fxc, **kwargs)
    return self._old_kernel(**kwargs)


@pyscf.lib.with_doc(TDA.kernel.__doc__)
def tda_kernel(self, **kwargs):
    '''Monkey-patched TDA kernel for CVS'''
    no_fxc, direct_diag = prepare_kernel(self, kwargs, core_valence)
    if direct_diag:
        return direct_diag_tda_kernel(self, no_fxc=no_fxc, **kwargs)
    return self._old_kernel(**kwargs)


patch_td_class(TDHF, rpa_kernel, core_valence)
patch_td_class(CasidaTDDFT, rpa_kernel, core_valence)
patch_td_class(TDA, tda_kernel, core_valence)
