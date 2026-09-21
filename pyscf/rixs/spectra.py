'''Help compute RIXS maps after peaks and heights have been assigned.'''

import numpy

from pyscf.data.nist import ALPHA
from pyscf.rixs.response import _normalize_state_indices


def select_significant_peaks(f_fn, threshold=1e-3):
    '''Return pair indices whose magnitude exceeds a relative threshold.

    Parameters
    ----------
    f_fn : array_like
        Pair factors with shape ``(nfinal, nintermediate)``.
    threshold : float, optional
        Minimum fraction of the largest pair-factor magnitude. Defaults to
        ``1e-3``.

    Returns
    -------
    numpy.ndarray
        Integer ``(final, intermediate)`` indices with shape ``(npeaks, 2)``.
    '''
    f_fn = numpy.asarray(f_fn)
    if f_fn.ndim != 2:
        raise ValueError('f_fn must be a two-dimensional array')
    if threshold < 0:
        raise ValueError('threshold must not be negative')

    maximum = numpy.max(numpy.abs(f_fn), initial=0)
    return numpy.argwhere(numpy.abs(f_fn) > threshold * maximum)


def rixs_map(f_fn, intermediate_energies, final_energies,
             incident_energy, transfer_energy, *, peaks=None,
             incident_broadening=1.0, transfer_broadening=1.0,
             alpha=ALPHA):
    '''Evaluate an averaged KH RIXS map.

    The returned map includes the kinematic outgoing-to-incoming photon
    energy factor ``(incident_energy - transfer_energy) / incident_energy``.

    Parameters
    ----------
    f_fn : array_like
        Pair factors with shape ``(nfinal, nintermediate)``.
    intermediate_energies : array_like
        Intermediate-state energies in the same units as the energy grids.
    final_energies : array_like
        Final-state energies in the same units as the energy grids.
    incident_energy : array_like
        One-dimensional incident-energy grid.
    transfer_energy : array_like
        One-dimensional energy-transfer grid.
    peaks : array_like of (int, int), optional
        ``(final, intermediate)`` pairs to include. Defaults to every
        nonzero pair in ``f_fn``.
    incident_broadening : float, optional
        Width in the incident-energy denominator.
    transfer_broadening : float, optional
        Broadening factor for the transfer-energy exponential used by the
        example workflow.
    alpha : float, optional
        Fine-structure constant used by the photon-energy prefactor.

    Returns
    -------
    X, Y, Z : ndarray
        Incident-energy mesh, transfer-energy mesh, and map intensity.
    '''
    f_fn = numpy.asarray(f_fn)
    intermediate_energies = numpy.asarray(intermediate_energies)
    final_energies = numpy.asarray(final_energies)
    incident_energy = numpy.asarray(incident_energy)
    transfer_energy = numpy.asarray(transfer_energy)

    if f_fn.ndim != 2:
        raise ValueError('f_fn must be a two-dimensional array')
    if intermediate_energies.ndim != 1:
        raise ValueError('intermediate_energies must be one-dimensional')
    if final_energies.ndim != 1:
        raise ValueError('final_energies must be one-dimensional')
    if f_fn.shape != (len(final_energies), len(intermediate_energies)):
        raise ValueError(
            'f_fn shape must be (nfinal, nintermediate)'
        )
    if incident_energy.ndim != 1:
        raise ValueError('incident_energy must be one-dimensional')
    if transfer_energy.ndim != 1:
        raise ValueError('transfer_energy must be one-dimensional')
    if incident_broadening <= 0:
        raise ValueError('incident_broadening must be positive')
    if transfer_broadening <= 0:
        raise ValueError('transfer_broadening must be positive')
    if numpy.any(incident_energy == 0):
        raise ValueError('incident_energy must not contain zero')

    X, Y = numpy.meshgrid(incident_energy, transfer_energy)
    Z = numpy.zeros_like(
        X,
        dtype=numpy.result_type(f_fn.dtype, numpy.float64),
    )
    photon_factor = (X - Y) / X

    if peaks is None:
        peaks = numpy.argwhere(f_fn != 0)
    else:
        peaks = numpy.asarray(peaks)
        if peaks.ndim != 2 or peaks.shape[1] != 2:
            raise ValueError('peaks must have shape (npeaks, 2)')
        f_positions = _normalize_state_indices(
            peaks[:, 0],
            len(final_energies),
            'peak final states',
        )
        n_positions = _normalize_state_indices(
            peaks[:, 1],
            len(intermediate_energies),
            'peak intermediate states',
        )
        peaks = numpy.column_stack((f_positions, n_positions))

    for f_pos, n_pos in peaks:
        wn = intermediate_energies[n_pos]
        wf = final_energies[f_pos]
        transfer_profile = numpy.exp(
            -5 / transfer_broadening * (Y - wf) ** 2
        )
        Z += (
            photon_factor
            * f_fn[f_pos, n_pos]
            * (wn * (wf - wn) * alpha) ** 2
            / (
                (X - wn) ** 2
                + .25 * incident_broadening ** 2
            )
            * transfer_profile
        )

    return X, Y, Z


def rixs_amplitudes(f_mu_n, n_mu_0, polarization_angle=0):
    '''Compute the RIXS amplitue for selected state pair.

    Computed using the Kramers-Heisenberg equation.

    Parameters
    ----------
    f_mu_n : array_like
        Intermediate-to-final transition dipole moment. Will be reshaped to
        (3, nfinal, nintermediate).
    n_mu_0 : array_like
        Ground-to-intermediate transition dipole moment. Will be reshaped to
        (3, nintermediate).
    polarization_angle : float
        Polarization angle of the incident X-ray beam relative to the measured.

    Returns
    -------
    float
        RIXS amplitude
    '''

    if polarization_angle != 0:
        raise NotImplementedError('Non-zero polarization angle is not yet supported')

    n_mu_0 = numpy.asarray(n_mu_0).reshape(3,-1)
    num_intermediate = n_mu_0.shape[1]
    f_mu_n = numpy.asarray(f_mu_n).reshape(3,-1,num_intermediate)

    s_fn = numpy.einsum(
        'xfn,yn->xyfn',
        f_mu_n.real,
        n_mu_0.real,
        optimize=True,
    )
    amplitudes = numpy.zeros(s_fn.shape[2:])
    for xi1 in range(3):
        for xi2 in range(3):
            amplitudes += (1 / 15) * (
                2 * s_fn[xi1, xi2] ** 2
                - .5 * (
                    s_fn[xi1, xi1] * s_fn[xi2, xi2]
                    + s_fn[xi1, xi2] * s_fn[xi2, xi1]
                )
            )
    return numpy.abs(amplitudes)
