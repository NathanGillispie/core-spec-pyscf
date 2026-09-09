'''CVS mixin for TDSCF objects.

Typical usage::

    import pyscf.cvs
    td = TDA(mf).cvs(core_idx=[0, 1, 2])
    td.kernel()

Assign the return value (``td = td.cvs(...)``). Importing this module attaches
``.cvs`` to :class:`pyscf.tdscf.rhf.TDBase`.
'''

from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import uhf as scf_uhf
from pyscf.scf import ghf as scf_ghf

from pyscf.cvs._utils import core_valence_restricted, core_valence_unrestricted
from pyscf.cvs.no_fxc import (
    get_ab_no_fxc_ghf,
    get_ab_no_fxc_rhf,
    get_ab_no_fxc_uhf,
)


def _mro_names(td):
    return {cls.__name__ for cls in type(td).__mro__}


def is_tda(td):
    '''True for Tamm–Dancoff objects; CasidaTDDFT is RPA despite subclassing TDA.'''
    names = _mro_names(td)
    if 'CasidaTDDFT' in names:
        return False
    return 'TDA' in names


def is_ghf_td(td):
    return isinstance(td._scf, scf_ghf.GHF)


def is_uhf_td(td):
    return isinstance(td._scf, scf_uhf.UHF) and not is_ghf_td(td)


def _get_ab(tdobj, no_fxc=False):
    frozen = getattr(tdobj, 'frozen', None)
    mf = tdobj._scf
    if no_fxc:
        if is_uhf_td(tdobj):
            return get_ab_no_fxc_uhf(mf, frozen=frozen)
        if is_ghf_td(tdobj):
            return get_ab_no_fxc_ghf(mf, frozen=frozen)
        return get_ab_no_fxc_rhf(mf, frozen=frozen)
    return tdobj.get_ab(mf=mf, frozen=frozen)


class CVS:
    '''Mixin that adds core-valence separation, no_fxc, and direct diagonalization.'''

    __name_mixin__ = 'CVS'
    _keys = {'core_idx', 'no_fxc', 'direct_diag'}

    def __init__(self, td, core_idx=None, no_fxc=False, direct_diag=False):
        self.__dict__.update(td.__dict__)
        self.core_idx = core_idx
        self.no_fxc = no_fxc
        self.direct_diag = direct_diag

    def undo_cvs(self):
        '''Remove the CVS mixin.'''
        obj = lib.view(self, lib.drop_class(self.__class__, CVS))
        for key in ('core_idx', 'no_fxc', 'direct_diag'):
            if hasattr(obj, key):
                delattr(obj, key)
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('core_idx = %s', self.core_idx)
        log.info('no_fxc = %s', self.no_fxc)
        log.info('direct_diag = %s', self.direct_diag)
        return self

    def core_valence(self, core_idx=None):
        if is_uhf_td(self):
            return core_valence_unrestricted(self, core_idx)
        return core_valence_restricted(self, core_idx)

    def get_ab(self, mf=None, frozen=None):
        if mf is None:
            mf = self._scf
        if frozen is None:
            frozen = self.frozen
        if is_ghf_td(self):
            from pyscf.tdscf.ghf import get_ab as ghf_get_ab
            return ghf_get_ab(mf, frozen=frozen)
        return super().get_ab(mf=mf, frozen=frozen)

    def kernel(self, x0=None, nstates=None, core_idx=None, no_fxc=None,
               direct_diag=None):
        if core_idx is None:
            core_idx = self.core_idx
        if no_fxc is None:
            no_fxc = self.no_fxc
        if direct_diag is None:
            direct_diag = self.direct_diag

        if core_idx is not None:
            self.core_valence(core_idx)
        if no_fxc and not direct_diag:
            logger.warn(self, 'No fxc requested. Using direct diagonalization.')
            direct_diag = True
        self.no_fxc = no_fxc
        self.direct_diag = direct_diag

        if direct_diag:
            return self._direct_diag_kernel(
                x0=x0, nstates=nstates, no_fxc=no_fxc)
        kwargs = {}
        if x0 is not None:
            kwargs['x0'] = x0
        if nstates is not None:
            kwargs['nstates'] = nstates
        return super().kernel(**kwargs)

    def _direct_diag_kernel(self, x0=None, nstates=None, no_fxc=False):
        tda = is_tda(self)
        if is_uhf_td(self):
            from pyscf.cvs import uhf as cvs_uhf
            fn = (cvs_uhf.direct_diag_tda_kernel
                  if tda else cvs_uhf.direct_diag_rpa_kernel)
        elif is_ghf_td(self):
            from pyscf.cvs import ghf as cvs_ghf
            fn = (cvs_ghf.direct_diag_tda_kernel
                  if tda else cvs_ghf.direct_diag_rpa_kernel)
        else:
            from pyscf.cvs import rhf as cvs_rhf
            fn = (cvs_rhf.direct_diag_tda_kernel
                  if tda else cvs_rhf.direct_diag_rpa_kernel)
        return fn(self, x0=x0, nstates=nstates, no_fxc=no_fxc)


def cvs(td, core_idx=None, no_fxc=None, direct_diag=None):
    '''Enable CVS / no_fxc / direct diagonalization on a TDSCF object.

    Args:
        td : a PySCF TDA, TDHF, TDDFT, or CasidaTDDFT object

    Kwargs:
        core_idx : occupied MO indices to excite from (UHF: ``(alpha, beta)``)
        no_fxc : bool
            Drop the XC kernel; always uses direct diagonalization.
        direct_diag : bool
            Diagonalize the A/B matrices with ``numpy.linalg.eigh``.

    Returns:
        The wrapped TD object. Assign the result: ``td = td.cvs(...)``.
    '''
    if isinstance(td, CVS):
        if core_idx is not None:
            td.core_valence(core_idx)
        if no_fxc is not None:
            td.no_fxc = no_fxc
        if direct_diag is not None:
            td.direct_diag = direct_diag
        return td

    obj = CVS(td,
              core_idx=core_idx,
              no_fxc=bool(no_fxc),
              direct_diag=bool(direct_diag))
    obj = lib.set_class(obj, (CVS, td.__class__))
    if core_idx is not None:
        obj.core_valence(core_idx)
    return obj
