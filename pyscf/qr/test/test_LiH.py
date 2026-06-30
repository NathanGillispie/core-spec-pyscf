'''Testing Transition Dipole Moments for LiH.'''

import numpy
import pytest
from pyscf import gto, lib, scf, dft
from pyscf.tdscf import RPA

from pyscf.qr import QR

@pytest.fixture
def lih_td():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='PBE0').run()
    td = RPA(mf).set(nstates=4)
    td.kernel()
    return td


def test_2tdm_smoke(lih_td):
    '''Making sure we can even run ``get_2tdm``.'''
    qr = QR(lih_td)

    qr.get_2tdm(0, 3)
    assert True


def test_2tdm(lih_td):
    qr = QR(lih_td)

    ref = numpy.asarray([
     [ 0.00000913, 0.00021478, 0.00196167,-0.        ,-0.        ,-0.00233548, 0.00059541, 0.00167035,-0.        ,-0.        , 0.00142593, 0.        ,-0.        , 0.00000221],
     [-0.00099577, 0.00138578, 0.05003507, 0.        , 0.        ,-0.02648603, 0.02447627, 0.00137882,-0.        ,-0.        ,-0.00290127,-0.        , 0.        ,-0.00126619],
     [ 0.00276826, 0.0529568 ,-0.06787215, 0.        , 0.        , 0.00927325,-0.00578   ,-0.0001081 , 0.        , 0.        , 0.00240265,-0.        , 0.        , 0.00002151],
     [-0.        ,-0.        , 0.        ,-0.        ,-0.        ,-0.        , 0.        , 0.        ,-0.        , 0.        ,-0.        , 0.        , 0.        ,-0.        ],
     [-0.        ,-0.        , 0.        ,-0.        ,-0.        ,-0.        , 0.        , 0.        ,-0.        , 0.        ,-0.        , 0.        , 0.        ,-0.        ],
     [-0.00218729, 0.21061183,-0.49008944, 0.        , 0.        , 0.06749361,-0.0423153 ,-0.00376811, 0.        ,-0.        , 0.01844034,-0.        ,-0.        , 0.0011216 ],
     [ 0.00071937, 0.02801048,-0.00645879,-0.        ,-0.        , 0.00095798,-0.00063134,-0.00042343, 0.        ,-0.        , 0.0003831 , 0.        ,-0.        , 0.00013768],
     [ 0.00135819,-0.00054368,-0.01278726,-0.        , 0.        , 0.00178108,-0.00112868,-0.00022737, 0.        ,-0.        , 0.00052508, 0.        ,-0.        , 0.00006889],
     [-0.        ,-0.        , 0.        ,-0.        ,-0.        ,-0.        , 0.        , 0.        ,-0.        , 0.        ,-0.        , 0.        , 0.        ,-0.        ],
     [-0.        ,-0.        , 0.        ,-0.        ,-0.        ,-0.        , 0.        , 0.        ,-0.        ,-0.        ,-0.        , 0.        , 0.        ,-0.        ],
     [ 0.0013693 ,-0.00652121, 0.00172117, 0.        , 0.        ,-0.00027759, 0.00019091, 0.00022488,-0.        , 0.        ,-0.00014656,-0.        , 0.        ,-0.00007505],
     [-0.        ,-0.        , 0.        ,-0.        ,-0.        ,-0.        , 0.        , 0.        ,-0.        ,-0.        ,-0.        , 0.        , 0.        ,-0.        ],
     [ 0.        , 0.        ,-0.        , 0.        , 0.        , 0.        ,-0.        ,-0.        , 0.        ,-0.        , 0.        ,-0.        ,-0.        , 0.        ],
     [-0.00001658,-0.00121688, 0.00254135, 0.        ,-0.        ,-0.00035327, 0.0002225 , 0.00003353,-0.        , 0.        ,-0.00010201,-0.        , 0.        ,-0.00001109]
    ])

    # Abs because sign randomly flips
    numpy.testing.assert_allclose(abs(ref), abs(qr.get_2tdm(0,3)), atol=1e-7)


# ---------------------------------------------------------------------------
# Step-by-step diagnostics: compare QR intermediates against the reference.
#
# These walk the pipeline C -> V -> Knm -> Pia/Qia so a failure pinpoints the
# exact step where the QR rewrite diverges from the reference math.  The "sum of
# absolute values" invariant guards against benign sign flips / degeneracies.
# ---------------------------------------------------------------------------

_PAIRS = [(0, 3), (0, 1), (1, 3), (2, 3)]


@pytest.mark.parametrize('pair', _PAIRS)
def test_intermediate_C(lih_intermediates, pair):
    '''C is independent of the state pair; it must match the reference.'''
    data = lih_intermediates[pair]
    ref_C = data['ref']['C']
    qr_C = data['qr']['C']
    numpy.testing.assert_allclose(_abssum(qr_C), _abssum(ref_C), rtol=1e-7)
    numpy.testing.assert_allclose(qr_C, ref_C, atol=1e-9)


@pytest.mark.parametrize('pair', _PAIRS)
def test_intermediate_V(lih_intermediates, pair):
    data = lih_intermediates[pair]
    ref_V = data['ref']['V']
    qr_V = data['qr']['V']
    numpy.testing.assert_allclose(_abssum(qr_V), _abssum(ref_V), rtol=1e-7)
    numpy.testing.assert_allclose(abs(qr_V), abs(ref_V), atol=1e-9)


@pytest.mark.parametrize('pair', _PAIRS)
def test_intermediate_Knm(lih_intermediates, pair):
    data = lih_intermediates[pair]
    ref_K = data['ref']['Knm']
    qr_K = data['qr']['Knm']
    numpy.testing.assert_allclose(_abssum(qr_K), _abssum(ref_K), rtol=1e-7)
    numpy.testing.assert_allclose(abs(qr_K), abs(ref_K), atol=1e-9)


@pytest.mark.parametrize('pair', _PAIRS)
def test_intermediate_Pia(lih_intermediates, pair):
    data = lih_intermediates[pair]
    ref_P = data['ref']['Pia']
    qr_P = data['qr']['Pia']
    numpy.testing.assert_allclose(_abssum(qr_P), _abssum(ref_P), rtol=1e-7)


@pytest.mark.parametrize('pair', _PAIRS)
def test_intermediate_Qia(lih_intermediates, pair):
    data = lih_intermediates[pair]
    ref_Q = data['ref']['Qia']
    qr_Q = data['qr']['Qia']
    numpy.testing.assert_allclose(_abssum(qr_Q), _abssum(ref_Q), rtol=1e-7)


# ---------------------------------------------------------------------------
# Transition dipole moment + oscillator strength (state 0 -> state 3).
#
# Hard-coded targets so the suite is self-contained.  Only the (0, 3) pair is
# used: LiH states 1 and 2 are degenerate (e = 0.18482305 Ha), so transitions
# involving them individually are not uniquely defined run-to-run.  States 0
# and 3 are non-degenerate, and the (0, 3) 2TDM is already validated
# element-wise by ``test_2tdm``.  The 2TDM sign flips between runs, so we test
# the (sign-robust) dipole magnitude.
# ---------------------------------------------------------------------------

_TDIP_MAG_03 = 0.15801795
_OSC_03 = 1.35903052e-03


def test_transition_dipole(lih_td):
    qr = QR(lih_td)
    tdm = qr.get_2tdm(0, 3)
    mag = float(numpy.linalg.norm(qr.transition_dipole(tdm)))
    numpy.testing.assert_allclose(mag, _TDIP_MAG_03, rtol=1e-7)


def test_oscillator_strength(lih_td):
    qr = QR(lih_td)
    numpy.testing.assert_allclose(
        qr.oscillator_strength(0, 3), _OSC_03, rtol=1e-7)


def test_oscillator_strength_tdm_passthrough(lih_td):
    '''Passing an explicit 2TDM must match the internally computed one.'''
    qr = QR(lih_td)
    tdm = qr.get_2tdm(0, 3)
    numpy.testing.assert_allclose(
        qr.oscillator_strength(0, 3, tdm=tdm),
        qr.oscillator_strength(0, 3))

