import pytest
import numpy
from pyscf.rixs import rixs_amplitudes

@pytest.fixture(scope='module')
def _state_pairs():
    '''Test data. Returns f_mu_n and n_mu_0 as tuple.'''
    fmn = numpy.arange(30).reshape(3,2,5)
    nm0 = numpy.arange(15).reshape(3,5)
    return fmn, nm0


def test_rixs_amplitudes(_state_pairs):
    f_mu_n, n_mu_0 = _state_pairs
    amp = rixs_amplitudes(f_mu_n, n_mu_0, polarization_angle=0)
    ref = numpy.array([[ 62500,  89104, 125104, 172444, 233284],
                       [113125, 155344, 211549, 284224, 376069]]) / 15.
    numpy.testing.assert_allclose(ref, amp)


def test_rixs_amplitudes_nonzero_polarization_nyi(_state_pairs):
    f_mu_n, n_mu_0 = _state_pairs
    with pytest.raises(NotImplementedError):
        rixs_amplitudes(f_mu_n, n_mu_0, polarization_angle=10)

