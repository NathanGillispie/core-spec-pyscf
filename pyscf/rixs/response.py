'''Reusable RIXS response helpers.'''

import numpy


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
