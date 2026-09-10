'''Shared helpers for the CVS plugin.'''

import numpy
from pyscf.lib import logger


def _as_index_list(idx):
    if isinstance(idx, (int, numpy.integer)):
        return [int(idx)]
    return [int(i) for i in numpy.atleast_1d(idx)]


def _as_energy_window(window):
    if not hasattr(window, '__len__') or len(window) != 2:
        raise ValueError('core_window must be a (emin, emax) pair')
    emin, emax = (float(x) for x in window)
    if numpy.isnan(emin) or numpy.isnan(emax) or emin > emax:
        raise ValueError('core_window must satisfy emin <= emax')
    return emin, emax


def _core_indices_from_window(mo_energy, mo_occ, window):
    if mo_energy is None:
        raise RuntimeError(
            'Orbital energies are required for core_window selection. '
            'Run the SCF calculation first.')
    mo_energy = numpy.asarray(mo_energy)
    mo_occ = numpy.asarray(mo_occ)
    if mo_energy.ndim != 1 or mo_occ.ndim != 1:
        raise ValueError('Restricted core_window selection requires 1D MO arrays')
    if mo_energy.size != mo_occ.size:
        raise ValueError('MO energy and occupation arrays have different sizes')

    emin, emax = _as_energy_window(window)
    occ_idx = numpy.where(mo_occ != 0)[0]
    core_idx = occ_idx[(mo_energy[occ_idx] >= emin)
                       & (mo_energy[occ_idx] <= emax)]
    if core_idx.size == 0:
        raise ValueError(
            f'core_window={window!r} does not contain an occupied orbital')
    return [int(i) for i in core_idx]


def _is_energy_window(window):
    return (hasattr(window, '__len__') and len(window) == 2
            and all(numpy.isscalar(x) for x in window))


def _unrestricted_energy_windows(window):
    if _is_energy_window(window):
        normalized = _as_energy_window(window)
        return normalized, normalized
    if (hasattr(window, '__len__') and len(window) == 2
            and all(_is_energy_window(x) for x in window)):
        return _as_energy_window(window[0]), _as_energy_window(window[1])
    raise ValueError(
        'UHF core_window must be (emin, emax) or '
        '((emin_alpha, emax_alpha), (emin_beta, emax_beta))')


def _core_indices_unrestricted_from_window(mo_energy, mo_occ, window):
    if mo_energy is None:
        raise RuntimeError(
            'Orbital energies are required for core_window selection. '
            'Run the SCF calculation first.')
    if len(mo_energy) != 2 or len(mo_occ) != 2:
        raise ValueError('Unrestricted core_window selection requires two spin arrays')
    windows = _unrestricted_energy_windows(window)
    return tuple(
        _core_indices_from_window(mo_energy[spin], mo_occ[spin], windows[spin])
        for spin in (0, 1))


def core_valence_restricted(tdobj, core_idx=None, core_window=None):
    '''Map core orbital indices onto ``tdobj.frozen`` for restricted/GHF refs.

    Occupied orbitals that are not listed in *core_idx* are frozen. The SCF
    ``mo_coeff`` / ``mo_occ`` / ``mo_energy`` arrays and ``mol.nelec`` are not
    modified.
    '''
    if core_idx is not None and core_window is not None:
        raise ValueError('Specify either core_idx or core_window, not both')
    if core_idx is None:
        if core_window is None:
            core_window = getattr(tdobj, 'core_window', None)
        if core_window is not None:
            core_idx = _core_indices_from_window(
                tdobj._scf.mo_energy, tdobj._scf.mo_occ, core_window)
        else:
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


def core_valence_unrestricted(tdobj, core_idx=None, core_window=None):
    '''Map per-spin core indices onto ``tdobj.frozen`` for UHF/UKS refs.

    *core_idx* must be ``(idx_alpha, idx_beta)``. Occupied orbitals of each
    spin that are not listed are frozen independently. Extra virtuals may be
    frozen on one spin so that both spins keep the same number of active MOs
    (required by PySCF's UHF Davidson solver). The SCF orbitals are not
    modified.
    '''
    if core_idx is not None and core_window is not None:
        raise ValueError('Specify either core_idx or core_window, not both')
    if core_idx is None:
        if core_window is None:
            core_window = getattr(tdobj, 'core_window', None)
        if core_window is not None:
            core_idx = _core_indices_unrestricted_from_window(
                tdobj._scf.mo_energy, tdobj._scf.mo_occ, core_window)
        else:
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
