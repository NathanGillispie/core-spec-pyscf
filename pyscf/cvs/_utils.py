'''Shared helpers for the CVS plugin.'''

import numpy
from pyscf.lib import logger


def parse_kernel_options(tdobj, kwargs):
    '''Pop CVS kernel options, falling back to attributes on *tdobj*.'''
    core_idx = kwargs.pop('core_idx', getattr(tdobj, 'core_idx', None))
    no_fxc = kwargs.pop('no_fxc', getattr(tdobj, 'no_fxc', False))
    direct_diag = kwargs.pop('direct_diag', getattr(tdobj, 'direct_diag', False))
    return core_idx, no_fxc, direct_diag


def _as_index_list(idx):
    if isinstance(idx, (int, numpy.integer)):
        return [int(idx)]
    return [int(i) for i in numpy.atleast_1d(idx)]


def core_valence_restricted(tdobj, core_idx=None):
    '''Map core orbital indices onto ``tdobj.frozen`` for restricted/GHF refs.

    Occupied orbitals that are not listed in *core_idx* are frozen. The SCF
    ``mo_coeff`` / ``mo_occ`` / ``mo_energy`` arrays and ``mol.nelec`` are not
    modified.
    '''
    if core_idx is None:
        core_idx = getattr(tdobj, 'core_idx', None)
    if core_idx is None:
        raise RuntimeError('Core orbitals not specified')

    tdobj.check_sanity()
    core_idx = _as_index_list(core_idx)
    occ_idx = numpy.where(tdobj._scf.mo_occ != 0)[0]
    if not numpy.all(numpy.isin(core_idx, occ_idx)):
        logger.warn(tdobj, 'Not all core orbitals are occupied!')

    frozen = numpy.setdiff1d(occ_idx, core_idx)
    tdobj.frozen = [int(i) for i in frozen]
    tdobj.core_idx = core_idx
    return tdobj


def _pad_frozen_uhf(mo_occ, frozen_a, frozen_b):
    '''Pad the shorter frozen list with virtuals so both spins have equal nmo.

    PySCF's UHF TDSCF Davidson solver assumes ``nmo_a == nmo_b`` after
    applying the frozen mask.
    '''
    frozen_a = [int(i) for i in frozen_a]
    frozen_b = [int(i) for i in frozen_b]
    vir_a = [int(i) for i in numpy.where(mo_occ[0] == 0)[0]
             if i not in frozen_a]
    vir_b = [int(i) for i in numpy.where(mo_occ[1] == 0)[0]
             if i not in frozen_b]
    n_a, n_b = len(frozen_a), len(frozen_b)
    if n_a < n_b:
        frozen_a.extend(vir_a[:n_b - n_a])
    elif n_b < n_a:
        frozen_b.extend(vir_b[:n_a - n_b])
    return frozen_a, frozen_b


def core_valence_unrestricted(tdobj, core_idx=None):
    '''Map per-spin core indices onto ``tdobj.frozen`` for UHF/UKS refs.

    *core_idx* must be ``(idx_alpha, idx_beta)``. Occupied orbitals of each
    spin that are not listed are frozen independently. Extra virtuals may be
    frozen on one spin so that both spins keep the same number of active MOs
    (required by PySCF's UHF Davidson solver). The SCF orbitals are not
    modified.
    '''
    if core_idx is None:
        core_idx = getattr(tdobj, 'core_idx', None)
    if core_idx is None:
        raise RuntimeError(
            'Core orbitals not specified. Use the core_idx attribute.')

    tdobj.check_sanity()
    if not hasattr(core_idx, '__len__') or len(core_idx) != 2:
        raise ValueError('core_idx must be in the form (idx_alpha, idx_beta)')

    core_a = _as_index_list(core_idx[0])
    core_b = _as_index_list(core_idx[1])
    occ_a = numpy.where(tdobj._scf.mo_occ[0] != 0)[0]
    occ_b = numpy.where(tdobj._scf.mo_occ[1] != 0)[0]
    if (not numpy.all(numpy.isin(core_a, occ_a))
            or not numpy.all(numpy.isin(core_b, occ_b))):
        logger.warn(tdobj, 'Not all core orbitals are occupied!')

    frozen_a = [int(i) for i in numpy.setdiff1d(occ_a, core_a)]
    frozen_b = [int(i) for i in numpy.setdiff1d(occ_b, core_b)]
    frozen_a, frozen_b = _pad_frozen_uhf(tdobj._scf.mo_occ, frozen_a, frozen_b)
    tdobj.frozen = (frozen_a, frozen_b)
    tdobj.core_idx = (core_a, core_b)
    return tdobj


def patch_td_class(cls, kernel, core_valence):
    '''Install CVS kernel/core_valence on *cls* without clobbering subclasses.'''
    if '_old_kernel' not in cls.__dict__:
        cls._old_kernel = cls.kernel
    cls.kernel = kernel
    cls.core_valence = core_valence
    extra = {'core_idx', 'no_fxc', 'direct_diag'}
    cls._keys = set(getattr(cls, '_keys', ())) | extra


def prepare_kernel(tdobj, kwargs, core_valence):
    '''Apply CVS options and decide whether to use direct diagonalization.'''
    core_idx, no_fxc, direct_diag = parse_kernel_options(tdobj, kwargs)
    if core_idx is not None:
        core_valence(tdobj, core_idx=core_idx)
    if no_fxc and not direct_diag:
        logger.warn(tdobj, 'No fxc requested. Using direct diagonalization.')
        direct_diag = True
    return no_fxc, direct_diag
