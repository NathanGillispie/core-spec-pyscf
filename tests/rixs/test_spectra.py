import pytest
import numpy
from pyscf.data.nist import ALPHA
from pyscf.rixs import (
    rixs_amplitudes,
    rixs_map,
    select_significant_peaks,
)

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


def test_select_significant_peaks():
    f_fn = numpy.array([[1.0, 10.0], [0.1, 5.0]])

    numpy.testing.assert_array_equal(
        select_significant_peaks(f_fn, threshold=.4),
        numpy.array([[0, 1], [1, 1]]),
    )


def test_select_significant_peaks_returns_empty_for_zero_input():
    numpy.testing.assert_array_equal(
        select_significant_peaks(numpy.zeros((2, 3))),
        numpy.empty((0, 2), dtype=int),
    )


def test_select_significant_peaks_rejects_invalid_input():
    with pytest.raises(ValueError):
        select_significant_peaks(numpy.ones(3))
    with pytest.raises(ValueError):
        select_significant_peaks(numpy.ones((2, 2)), threshold=-1)


def test_rixs_map():
    X, Y, Z = rixs_map(
        numpy.array([[2.0]]),
        numpy.array([3.0]),
        numpy.array([1.0]),
        numpy.array([2.0]),
        numpy.array([1.0]),
    )

    expected = .5 * 2 * (3 * (1 - 3) * ALPHA) ** 2 / (1 + .25)
    assert X.shape == (1, 1)
    assert Y.shape == (1, 1)
    numpy.testing.assert_allclose(Z, expected)


def test_rixs_map_rejects_out_of_range_peaks():
    with pytest.raises(IndexError, match='peak final states'):
        rixs_map(
            numpy.array([[2.0]]),
            numpy.array([3.0]),
            numpy.array([1.0]),
            numpy.array([2.0]),
            numpy.array([1.0]),
            peaks=[(1, 0)],
        )

