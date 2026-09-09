import pyscf
from pyscf.tdscf.rhf import TDHF, TDA
from pyscf.tdscf.rks import CasidaTDDFT
import numpy
from scipy.linalg import sqrtm

from pyscf.lib import logger
from .no_fxc import get_ab_no_fxc_rhf
from ._utils import core_valence_restricted, patch_td_class, prepare_kernel

core_valence = core_valence_restricted


def _get_ab(tdobj, no_fxc=False):
    frozen = getattr(tdobj, 'frozen', None)
    mf = tdobj._scf
    if no_fxc:
        return get_ab_no_fxc_rhf(mf, frozen=frozen)
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

    A, _ = _get_ab(self, no_fxc=no_fxc)
    assert A.dtype == numpy.float64
    nocc = A.shape[0]
    nvir = A.shape[1]
    A = A.reshape(nocc * nvir, nocc * nvir)

    e, x1 = numpy.linalg.eigh(A)

    keep_idx = numpy.where(e > self.positive_eig_threshold)[0]
    e = e[keep_idx]
    x1 = x1[:, keep_idx]

    self.e = e[:nstates]
    x1 = x1[:, :nstates]

    self.xy = [(xi.reshape(nocc, nvir) * numpy.sqrt(.5), 0) for xi in x1.T]
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

    A, B = _get_ab(self, no_fxc=no_fxc)
    assert A.dtype == numpy.float64
    nocc = A.shape[0]
    nvir = A.shape[1]
    A = A.reshape(nocc * nvir, nocc * nvir)
    B = B.reshape(nocc * nvir, nocc * nvir)

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
    x1 = numpy.zeros((X.shape[0] * 2, X.shape[1]))
    x1[:nocc * nvir] += X
    x1[nocc * nvir:] += Y

    keep_idx = numpy.where(e > self.positive_eig_threshold)[0]

    def norm_xy(z):
        x, y = z.reshape(2, -1)
        norm = pyscf.lib.norm(x)**2 - pyscf.lib.norm(y)**2
        if norm < 0:
            log.warn('TDDFT amplitudes |X| smaller than |Y|')
        norm = abs(.5 / norm)**.5  # normalize to 0.5 for alpha spin
        return x.reshape(nocc, nvir) * norm, y.reshape(nocc, nvir) * norm

    xy = [norm_xy(z) for i, z in enumerate(x1.T) if i in keep_idx]

    self.xy = xy[:nstates]
    self.e = (e[keep_idx])[:nstates]
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
