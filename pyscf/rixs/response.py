'''Reusable RIXS response helpers.'''

import numpy

from pyscf.data.nist import HARTREE2EV
from pyscf.qr.dipole import compute_dipole_mo


def _normalize_state_indices(states, size, name):
    if states is None:
        return numpy.arange(size, dtype=int)

    states = numpy.atleast_1d(numpy.asarray(states))
    if states.ndim != 1:
        raise ValueError(f'{name} must be a one-dimensional array')
    if not numpy.issubdtype(states.dtype, numpy.integer):
        raise ValueError(f'{name} must contain integer state indices')

    states = states.astype(int, copy=False)
    if numpy.any(states < 0) or numpy.any(states >= size):
        raise IndexError(
            f'{name} contains an index outside [0, {size})'
        )
    return states


def _to_hartree(energies, unit):
    try:
        unit = unit.lower()
    except AttributeError as err:
        raise ValueError(f'unsupported energy unit: {unit!r}') from err

    if unit in ('au', 'ha', 'hartree'):
        return energies
    if unit == 'ev':
        return energies / HARTREE2EV
    raise ValueError(f'unsupported energy unit: {unit!r}')


def select_states(energies, window, *, energies_unit='au',
                  window_unit=None):
    '''Return indices of states within an inclusive energy window.

    Parameters
    ----------
    energies : array_like
        One-dimensional state energies.
    window : 2-tuple
        Lower and upper bounds.
    energies_unit : {'au', 'ha', 'hartree', 'eV'}, optional
        Units of ``energies``. Defaults to Hartree atomic units.
    window_unit : {'au', 'ha', 'hartree', 'eV'}, optional
        Units of ``window``. Defaults to ``energies_unit``.

    Returns
    -------
    numpy.ndarray
        Integer indices of states satisfying
        ``window[0] <= energies <= window[1]``.
    '''
    energies = numpy.asarray(energies)
    if energies.ndim != 1:
        raise ValueError('energies must be a one-dimensional array')

    try:
        low, high = window
    except (TypeError, ValueError) as err:
        raise ValueError('window must contain exactly two bounds') from err

    if window_unit is None:
        window_unit = energies_unit
    energies = _to_hartree(energies, energies_unit)
    low, high = _to_hartree(
        numpy.asarray((low, high), dtype=float),
        window_unit,
    )
    if low > high:
        raise ValueError('window lower bound must not exceed upper bound')

    return numpy.flatnonzero((energies >= low) & (energies <= high))


def ground_transition_dipoles(mol, mo_coeff, mo_occ, manifold, states=None,
                              dipole_mo=None):
    '''Compute ground-to-excited transition dipoles from a LR manifold.

    Parameters
    ----------
    mol : Mole
        Molecular object used to evaluate the length-gauge dipole integrals.
    mo_coeff : ndarray
        Molecular-orbital coefficients.
    mo_occ : ndarray
        Molecular-orbital occupations.
    manifold : Manifold
        Linear-response manifold containing the excited-state amplitudes.
    states : array_like of int, optional
        0-based state indices.  Defaults to all states in ``manifold``.
    dipole_mo : ndarray, optional
        Precomputed length-gauge dipole integrals in the MO basis.  When
        omitted, they are computed from ``mol`` and ``mo_coeff``.

    Returns
    -------
    numpy.ndarray
        Transition dipoles with shape ``(3, len(states))``.

    Notes
    -----
    This is the closed-shell expression used by PySCF's restricted
    linear-response transition-dipole implementation.  The ``X + Y``
    transition density supports both TDA and RPA manifolds.
    '''
    if states is None:
        states = numpy.arange(len(manifold.e))
    else:
        states = numpy.atleast_1d(states)
    if states.ndim != 1:
        raise ValueError('states must be a one-dimensional array')

    mo_occ = numpy.asarray(mo_occ)
    mo_coeff = numpy.asarray(mo_coeff)
    occ_idx = numpy.flatnonzero(mo_occ == 2)
    virt_idx = numpy.flatnonzero(mo_occ == 0)

    if dipole_mo is None:
        dipole_mo = compute_dipole_mo(mol, mo_coeff)
    dipole_mo = numpy.asarray(dipole_mo)
    dip_ov = dipole_mo[:, occ_idx][:, :, virt_idx]

    dipoles = []
    for state in states:
        _, (x, y) = manifold(int(state))
        dipoles.append(2 * numpy.einsum('xia,ia->x', dip_ov, x + y))

    if not dipoles:
        dtype = numpy.result_type(dip_ov, mo_coeff)
        return numpy.empty((3, 0), dtype=dtype)
    return numpy.asarray(dipoles).T
