import numpy
import pytest

from pyscf.rixs import select_states


def test_select_states_includes_window_boundaries():
    energies = numpy.array([0.5, 1.0, 2.0, 3.0, 3.5])

    numpy.testing.assert_array_equal(
        select_states(energies, (1.0, 3.0)),
        numpy.array([1, 2, 3]),
    )


def test_select_states_returns_empty_for_no_matches():
    numpy.testing.assert_array_equal(
        select_states(numpy.array([1.0, 2.0]), (3.0, 4.0)),
        numpy.array([], dtype=int),
    )


@pytest.mark.parametrize(
    'energies, window',
    [
        (numpy.array([[1.0]]), (0.0, 2.0)),
        (numpy.array([1.0]), (2.0, 0.0)),
        (numpy.array([1.0]), (0.0, 1.0, 2.0)),
    ],
)
def test_select_states_rejects_invalid_input(energies, window):
    with pytest.raises(ValueError):
        select_states(energies, window)
