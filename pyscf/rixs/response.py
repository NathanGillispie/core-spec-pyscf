'''Reusable RIXS response helpers.'''

import numpy


def _charge_center(mol):
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    return numpy.einsum('z,zr->r', charges, coords) / charges.sum()


def select_states(energies, window):
    '''Return indices of states within an inclusive energy window.

    Parameters
    ----------
    energies : array_like
        One-dimensional state energies.
    window : 2-tuple
        Lower and upper bounds in the same units as ``energies``.  PySCF
        energies are normally expressed in Hartree.

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

    if low > high:
        raise ValueError('window lower bound must not exceed upper bound')

    return numpy.flatnonzero((energies >= low) & (energies <= high))


def ground_transition_dipoles(mol, mo_coeff, mo_occ, manifold, states=None):
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

    with mol.with_common_orig(_charge_center(mol)):
        dip_ao = mol.intor_symmetric('int1e_r', comp=3)
    dip_ov = numpy.einsum(
        'xpq,pi,qa->xia',
        dip_ao,
        mo_coeff[:, occ_idx],
        mo_coeff[:, virt_idx],
    )

    dipoles = []
    for state in states:
        _, (x, y) = manifold(int(state))
        dipoles.append(2 * numpy.einsum('xia,ia->x', dip_ov, x + y))

    if not dipoles:
        dtype = numpy.result_type(dip_ov, mo_coeff)
        return numpy.empty((3, 0), dtype=dtype)
    return numpy.asarray(dipoles).T
