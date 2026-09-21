'''CVS mixin for TDSCF objects.

Typical usage::

    import pyscf.cvs
    td = TDA(mf).cvs(core_idx=[0, 1, 2])
    td.kernel()

An inclusive MO-energy window can be used instead of explicit indices::

    td = TDA(mf).cvs(core_window=(-20.0, -10.0))

Assign the return value (``td = td.cvs(...)``). Importing this module attaches
``.cvs`` to :class:`pyscf.tdscf.rhf.TDBase`.
'''

import copy

from pyscf import lib, scf
from pyscf.lib import logger
from pyscf.scf import uhf as scf_uhf
from pyscf.scf import ghf as scf_ghf

from pyscf.cvs._utils import core_valence_restricted, core_valence_unrestricted


def _mro_names(td):
    return {cls.__name__ for cls in type(td).__mro__}


def is_tda(td):
    '''True for Tamm–Dancoff objects; CasidaTDDFT is RPA despite subclassing TDA.'''
    names = _mro_names(td)
    if 'CasidaTDDFT' in names:
        return False
    return 'TDA' in names


def is_casida(td):
    return 'CasidaTDDFT' in _mro_names(td)


def is_ghf_td(td):
    return isinstance(td._scf, scf_ghf.GHF)


def is_uhf_td(td):
    return isinstance(td._scf, scf_uhf.UHF) and not is_ghf_td(td)


def _get_no_fxc_mf(tdobj):
    '''Return a response-only mean-field object for no_fxc calculations.'''
    mf = tdobj._scf
    if not isinstance(mf, scf.hf.KohnShamDFT):
        return mf
    return mf.to_hf()


def _get_ab(tdobj, no_fxc=False):
    frozen = getattr(tdobj, 'frozen', None)
    mf = _get_no_fxc_mf(tdobj) if no_fxc else tdobj._scf
    return tdobj.get_ab(mf=mf, frozen=frozen)


class CVS:
    '''Mixin that adds core-valence separation, no_fxc, and direct diagonalization.'''

    __name_mixin__ = 'CVS'
    _keys = {'core_idx', 'core_window', 'no_fxc', 'direct_diag'}

    def __init__(self, td, core_idx=None, core_window=None, no_fxc=False,
                 direct_diag=False):
        self.__dict__.update(td.__dict__)
        self.core_idx = core_idx
        self.core_window = core_window
        self.no_fxc = no_fxc
        self.direct_diag = direct_diag

    def undo_cvs(self):
        '''Remove the CVS mixin.'''
        obj = lib.view(self, lib.drop_class(self.__class__, CVS))
        for key in ('core_idx', 'core_window', 'no_fxc', 'direct_diag'):
            if hasattr(obj, key):
                delattr(obj, key)
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('core_idx = %s', self.core_idx)
        log.info('core_window = %s', self.core_window)
        log.info('no_fxc = %s', self.no_fxc)
        log.info('direct_diag = %s', self.direct_diag)
        return self

    def core_valence(self, core_idx=None, core_window=None):
        if core_idx is not None:
            self.core_window = None
        elif core_window is not None:
            self.core_window = core_window
            self.core_idx = None
        if is_uhf_td(self):
            return core_valence_unrestricted(self, core_idx, core_window)
        return core_valence_restricted(self, core_idx, core_window)

    def get_ab(self, mf=None, frozen=None):
        if mf is None:
            mf = _get_no_fxc_mf(self) if self.no_fxc else self._scf
        if frozen is None:
            frozen = self.frozen
        if is_ghf_td(self):
            from pyscf.tdscf.ghf import get_ab as ghf_get_ab
            return ghf_get_ab(mf, frozen=frozen)
        return super().get_ab(mf=mf, frozen=frozen)

    def gen_response(self, *args, **kwargs):
        if not self.no_fxc:
            return super().gen_response(*args, **kwargs)
        kwargs = dict(kwargs)
        kwargs['with_nlc'] = False
        return _get_no_fxc_mf(self).gen_response(*args, **kwargs)

    def gen_vind(self, mf=None):
        if not self.no_fxc:
            return super().gen_vind(mf)

        response_td = copy.copy(self)
        response_td._scf = _get_no_fxc_mf(self)
        response_td.no_fxc = False
        if is_casida(self):
            if is_uhf_td(self):
                from pyscf.tdscf.uhf import TDHF
            elif is_ghf_td(self):
                from pyscf.tdscf.ghf import TDHF
            else:
                from pyscf.tdscf.rhf import TDHF
            return TDHF.gen_vind(response_td, response_td._scf)
        return super(CVS, response_td).gen_vind(response_td._scf)

    def get_init_guess(self, mf, nstates=None, wfnsym=None,
                       return_symmetry=False):
        if self.no_fxc and is_casida(self):
            if is_uhf_td(self):
                from pyscf.tdscf.uhf import TDHF
            elif is_ghf_td(self):
                from pyscf.tdscf.ghf import TDHF
            else:
                from pyscf.tdscf.rhf import TDHF
            return TDHF.get_init_guess(
                self, mf, nstates, wfnsym, return_symmetry)
        return super().get_init_guess(
            mf, nstates, wfnsym, return_symmetry)

    def kernel(self, x0=None, nstates=None, core_idx=None, core_window=None,
               no_fxc=None, direct_diag=None):
        if core_idx is not None and core_window is not None:
            raise ValueError('Specify either core_idx or core_window, not both')
        if core_idx is not None:
            self.core_window = None
        elif core_window is not None:
            self.core_window = core_window
        elif core_window is None:
            core_window = self.core_window
        if core_window is not None:
            core_idx = None
        elif core_idx is None:
            core_idx = self.core_idx
        if no_fxc is None:
            no_fxc = self.no_fxc
        if direct_diag is None:
            direct_diag = self.direct_diag

        if core_window is not None:
            self.core_valence(core_window=core_window)
        elif core_idx is not None:
            self.core_valence(core_idx)
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
        if no_fxc and is_casida(self):
            if is_uhf_td(self):
                from pyscf.tdscf.uhf import TDHF
            elif is_ghf_td(self):
                from pyscf.tdscf.ghf import TDHF
            else:
                from pyscf.tdscf.rhf import TDHF
            return TDHF.kernel(self, **kwargs)
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


def cvs(td, core_idx=None, core_window=None, no_fxc=None, direct_diag=None):
    '''Enable CVS / no_fxc / direct diagonalization on a TDSCF object.

    Args:
        td : a PySCF TDA, TDHF, TDDFT, or CasidaTDDFT object

    Kwargs:
        core_idx : occupied MO indices to excite from (UHF: ``(alpha, beta)``)
        core_window : ``(emin, emax)`` energy window for occupied core MOs
            (UHF also accepts separate alpha and beta windows)
        no_fxc : bool
            No exchange-correlation contributions; converts KS to HF without
            modifying the mean-field or td object.
        direct_diag : bool
            Diagonalize the A/B matrices with ``numpy.linalg.eigh``.

    Returns:
        The wrapped TD object. Assign the result: ``td = td.cvs(...)``.
    '''
    if core_idx is not None and core_window is not None:
        raise ValueError('Specify either core_idx or core_window, not both')
    if isinstance(td, CVS):
        if core_idx is not None:
            td.core_valence(core_idx)
        elif core_window is not None:
            td.core_window = core_window
            td.core_idx = None
        if no_fxc is not None:
            td.no_fxc = no_fxc
        if direct_diag is not None:
            td.direct_diag = direct_diag
        return td

    obj = CVS(td,
              core_idx=core_idx,
              core_window=core_window,
              no_fxc=bool(no_fxc),
              direct_diag=bool(direct_diag))
    obj = lib.set_class(obj, (CVS, td.__class__))
    if core_idx is not None:
        obj.core_valence(core_idx)
    return obj
