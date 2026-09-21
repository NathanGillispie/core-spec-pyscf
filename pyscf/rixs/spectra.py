'''Help compute RIXS maps after peaks and heights have been assigned.'''

import numpy


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
