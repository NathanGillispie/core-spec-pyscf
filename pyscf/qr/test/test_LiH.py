'''Testing Transition Dipole Moments for LiH.'''

import os
import numpy
import pytest
from pyscf import gto, lib, scf, dft
from pyscf.tdscf import RPA

from pyscf.qr import QR
from pyscf.qr.rhf import (
    _compute_c,
    _get_2tdm_diag_block,
    _get_pq,
    LazyGxc,
)

# ---------------------------------------------------------------------------
# Step-by-step check for QR intermediates against a reference.
#
# These walk through the computation of each C, V, Knm, Pia, Qia.
# Since we hard code the SCF results, everything is deterministic + we don't
# have to worry about sign flips / degeneracies flipping MOs.
# ---------------------------------------------------------------------------

_DATA = os.path.join(os.path.dirname(__file__), 'lih_ref_intermediates.npz')
_PAIRS = [(0, 3), (0, 1), (1, 3), (2, 3)]

@pytest.fixture(scope='module')
def lih_intermediates():
    '''Stored-reference + freshly-computed QR intermediates for LiH state pairs.

    The MO basis is pinned to the stored one and the QR path is deteministic.
    '''

    # Get pre-computed reference intermediates 
    data = numpy.load(_DATA)

    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='PBE0').run()

    # Replace the MO basis so QR results are directly comparable
    mf.mo_coeff = data['mo_coeff']
    mf.mo_energy = data['mo_energy']
    mf.mo_occ = data['mo_occ']

    out = {}
    backend = LazyGxc()
    C_qr = _compute_c(mf)
    for (n, m) in _PAIRS:
        x1, y1 = data[f'x{n}'], data[f'y{n}']
        x2, y2 = data[f'x{m}'], data[f'y{m}']

        V_qr = backend.contract_v(mf, x1 + y1, x2 + y2)
        Knm_qr = _get_2tdm_diag_block(x1, x2, y1, y2)
        Pia_qr, Qia_qr = _get_pq(C_qr, Knm_qr, V_qr, x1, x2, y1, y2)

        out[(n, m)] = dict(
            ref=dict(
                C=data['C'], V=data[f'V_{n}_{m}'], Knm=data[f'Knm_{n}_{m}'],
                Pia=data[f'Pia_{n}_{m}'], Qia=data[f'Qia_{n}_{m}'],
            ),
            qr=dict(C=C_qr, V=V_qr, Knm=Knm_qr, Pia=Pia_qr, Qia=Qia_qr),
        )
    return out


@pytest.mark.parametrize('pair', _PAIRS)
def test_intermediate_C(lih_intermediates, pair):
    '''C is independent of the state pair; it must match the reference.'''
    data = lih_intermediates[pair]
    ref_C = data['ref']['C']
    qr_C = data['qr']['C']
    numpy.testing.assert_allclose(qr_C, ref_C, atol=1e-9)


@pytest.mark.parametrize('pair', _PAIRS)
def test_intermediate_V(lih_intermediates, pair):
    data = lih_intermediates[pair]
    ref_V = data['ref']['V']
    qr_V = data['qr']['V']
    numpy.testing.assert_allclose(abs(qr_V), abs(ref_V), atol=1e-9)
    numpy.testing.assert_allclose(qr_V, ref_V, atol=1e-9)


@pytest.mark.parametrize('pair', _PAIRS)
def test_intermediate_Knm(lih_intermediates, pair):
    data = lih_intermediates[pair]
    ref_K = data['ref']['Knm']
    qr_K = data['qr']['Knm']
    numpy.testing.assert_allclose(abs(qr_K), abs(ref_K), atol=1e-9)
    numpy.testing.assert_allclose(qr_K, ref_K, atol=1e-9)


@pytest.mark.parametrize('pair', _PAIRS)
def test_intermediate_Pia(lih_intermediates, pair):
    data = lih_intermediates[pair]
    ref_P = data['ref']['Pia']
    qr_P = data['qr']['Pia']
    numpy.testing.assert_allclose(qr_P, ref_P, atol=1e-9)


@pytest.mark.parametrize('pair', _PAIRS)
def test_intermediate_Qia(lih_intermediates, pair):
    data = lih_intermediates[pair]
    ref_Q = data['ref']['Qia']
    qr_Q = data['qr']['Qia']
    numpy.testing.assert_allclose(qr_Q, ref_Q, atol=1e-9)


# ---------------------------------------------------------------------------
#   Transition dipole moment + oscillator strength (state 0 -> state 3).    |
# ---------------------------------------------------------------------------

@pytest.fixture
def lih_td():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='PBE0').run()
    td = RPA(mf).set(nstates=4)
    td.kernel()
    return td


def test_oscillator_strength_tdm_passthrough(lih_td):
    '''Passing an explicit 2TDM must match the internally computed one.'''
    qr = QR(lih_td)
    tdm = qr.get_2tdm(0, 3)
    numpy.testing.assert_allclose(
        qr.oscillator_strength(0, 3, tdm=tdm),
        qr.oscillator_strength(0, 3))

def test_transition_dipole(lih_td):
    qr = QR(lih_td)
    tdm = qr.get_2tdm(0, 3)
    mag = float(numpy.linalg.norm(qr.transition_dipole(tdm)))
    numpy.testing.assert_allclose(mag, 0.15801795, rtol=1e-7)


def test_oscillator_strength(lih_td):
    qr = QR(lih_td)
    numpy.testing.assert_allclose(
        qr.oscillator_strength(0, 3), 1.35903052e-03, rtol=1e-7)


