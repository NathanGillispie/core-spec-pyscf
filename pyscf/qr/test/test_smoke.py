'''Smoke tests for the pyscf.qr skeleton.'''

import json

import numpy
import pytest
from pyscf import gto, lib, scf
from pyscf.tdscf import RPA, TDA

import pyscf.qr
from pyscf.qr import Manifold, QR
from pyscf.qr.manifold import gxc_tensor_shape
from pyscf.qr.rhf import EagerGxc
from pyscf.qr.uhf import UQR
from pyscf.qr.ghf import GQR


def test_plugin_import():
    import pyscf.qr


def test_public_exports():
    assert hasattr(pyscf.qr, 'Manifold')
    assert hasattr(pyscf.qr, 'QR')
    assert hasattr(pyscf.qr, 'UQR')
    assert hasattr(pyscf.qr, 'GQR')
    assert hasattr(pyscf.qr, 'gxc_tensor_shape')


@pytest.fixture
def he_mf():
    mol = gto.M(atom='He 0 0 0', basis='6-31g', verbose=0)
    return scf.RHF(mol).run()


@pytest.fixture
def h2_mf():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g', verbose=0)
    return scf.RHF(mol).run()

def test_manifold_loads_legacy_ndarray(he_mf):
    legacy_rpa = json.dumps({
        'occ_idx': [0],
        'e': [0.1],
        'xy': numpy.zeros((1, 2, 1, 1)).tolist(),
    })
    manifold = Manifold.loads(legacy_rpa, he_mf)
    assert manifold.xy[0][1] is not None

    legacy_tda = json.dumps({
        'occ_idx': [0],
        'e': [0.1],
        'xy': numpy.zeros((1, 1, 1)).tolist(),
    })
    manifold = Manifold.loads(legacy_tda, he_mf)
    assert manifold.xy[0][1] is None


def test_qr_construction_from_tdobj(he_mf):
    td = RPA(he_mf).set(nstates=1)
    td.kernel()
    qr = QR(td)
    assert len(qr.manifold_n.e) == 1
    assert qr.manifold_m is qr.manifold_n
    assert qr.response_type == 'rpa'
    assert qr.mo_coeff is he_mf.mo_coeff
    assert qr.mol is he_mf.mol
    assert qr.manifold_n.mo_coeff is he_mf.mo_coeff


def test_qr_runs_lr_if_needed(he_mf):
    td = RPA(he_mf).set(nstates=1)
    qr = QR(td)
    assert len(qr.manifold_n.e) == 1


def test_qr_two_manifolds_shared_reference(he_mf):
    td1 = RPA(he_mf).set(nstates=1)
    td1.kernel()
    td2 = RPA(he_mf, frozen=0).set(nstates=1)
    td2.kernel()
    qr = QR(td1, td2)
    assert qr.manifold_n is not qr.manifold_m
    assert len(qr.manifold_n.e) == 1
    assert len(qr.manifold_m.e) == 1


def test_qr_rejects_tda_rpa_mix(he_mf):
    td_tda = TDA(he_mf).set(nstates=1)
    td_tda.kernel()
    td_rpa = RPA(he_mf).set(nstates=1)
    td_rpa.kernel()
    with pytest.raises(ValueError, match='same response type'):
        QR(td_tda, td_rpa)


def test_manifold_occ_idx(he_mf):
    td = RPA(he_mf).set(nstates=1)
    td.kernel()
    manifold = Manifold.from_tdobj(td)
    assert len(manifold.occ_idx) == 1
    assert manifold.occ_idx[0] == 0


def test_gxc_tensor_shape_two_manifolds(he_mf):
    occ_a = numpy.array([0], dtype=int)
    occ_b = numpy.array([0, 1], dtype=int)
    man_a = Manifold(
        mol=he_mf.mol, mo_coeff=he_mf.mo_coeff, mo_occ=he_mf.mo_occ,
        occ_idx=occ_a,
        e=numpy.array([0.1]),
        xy=((numpy.zeros((1, 1)), numpy.zeros((1, 1))),),
    )
    man_b = Manifold(
        mol=he_mf.mol, mo_coeff=he_mf.mo_coeff, mo_occ=he_mf.mo_occ,
        occ_idx=occ_b,
        e=numpy.array([0.2]),
        xy=((numpy.zeros((2, 1)), numpy.zeros((2, 1))),),
    )
    nvirt = int(numpy.count_nonzero(he_mf.mo_occ == 0))
    shape = gxc_tensor_shape(man_a, man_b, nvirt)
    assert shape == (1, nvirt, 1, 1, 2, 1)


def test_eager_gxc_shares_buffer(he_mf):
    td = RPA(he_mf).set(nstates=1)
    td.kernel()
    qr = QR(td, precompute_gxc=True)
    assert isinstance(qr._gxc_backend, EagerGxc)
    assert qr._gxc_backend.G is qr._gxc


def test_qr_get_2tdm_smoke(h2_mf):
    td = RPA(h2_mf).set(nstates=2)
    td.kernel()
    qr = QR(td)

    tdm = qr.get_2tdm(0, 1)
    nmo = h2_mf.mo_coeff.shape[1]
    assert tdm.shape == (nmo, nmo)


@pytest.mark.parametrize('approximation', ['Nascimento', 'Zero', 'Pseudo'])
def test_qr_get_2tdm_approximation(h2_mf, approximation):
    td = RPA(h2_mf).set(nstates=2)
    td.kernel()
    qr = QR(td, approximation=approximation)

    tdm = qr.get_2tdm(0, 1)
    nmo = h2_mf.mo_coeff.shape[1]
    assert tdm.shape == (nmo, nmo)


def test_uqr_not_implemented():
    with pytest.raises(NotImplementedError):
        UQR(None)


def test_gqr_not_implemented():
    with pytest.raises(NotImplementedError):
        GQR(None)


def test_qr_save_and_from_chk(h2_mf, tmp_path):
    td = RPA(h2_mf).set(nstates=1)
    td.kernel()
    chk = str(tmp_path / 'qr.chk')
    qr = QR(td, chkfile=chk)
    qr.save()

    with lib.H5FileWrap(chk, 'r') as f:
        assert 'qr/manifold_n' in f

    qr2 = QR.from_chk(chk, h2_mf)
    assert type(qr2).__name__ == 'RQR'
    numpy.testing.assert_array_equal(qr2.manifold_n.e, qr.manifold_n.e)
    assert qr2.response_type == 'rpa'
    assert qr2.manifold_n.mol is h2_mf.mol
    assert qr2.manifold_n.mo_coeff is h2_mf.mo_coeff
    assert qr2.mo_coeff is h2_mf.mo_coeff
    assert qr2.kernel() is qr2


def test_from_chk_requires_mf(h2_mf, tmp_path):
    td = RPA(h2_mf).set(nstates=1)
    td.kernel()
    chk = str(tmp_path / 'qr.chk')
    QR(td, chkfile=chk).save()

    with pytest.raises(TypeError):
        QR.from_chk(chk)


def test_get_aligned_xy_padding(he_mf):
    mo_occ = numpy.array([2, 2, 0, 0, 0, 0])
    occ_idx = numpy.array([0], dtype=int)
    nvirt = int(numpy.count_nonzero(mo_occ == 0))
    man = Manifold(
        mol=he_mf.mol, mo_coeff=he_mf.mo_coeff, mo_occ=mo_occ,
        occ_idx=occ_idx,
        e=numpy.array([0.1]), xy=((numpy.ones((1, nvirt)), None),),
    )
    x_pad, y_pad = man.get_aligned_xy(0)
    assert x_pad.shape == (2, nvirt)
    assert y_pad.shape == (2, nvirt)
    assert x_pad[0, 0] == 1.0
    assert x_pad[1, 0] == 0.0


def test_manifold_call(he_mf):
    mo_occ = numpy.array([2, 2, 0, 0, 0, 0])
    occ_idx = numpy.array([0], dtype=int)
    nvirt = int(numpy.count_nonzero(mo_occ == 0))
    e = numpy.array([0.42])
    man = Manifold(
        mol=he_mf.mol, mo_coeff=he_mf.mo_coeff, mo_occ=mo_occ,
        occ_idx=occ_idx,
        e=e, xy=((numpy.ones((1, nvirt)), None),),
    )
    e_out, (x, y) = man(0)
    assert e_out == 0.42
    assert x.shape == (2, nvirt)
    assert y.shape == (2, nvirt)


def test_manifold_dump_roundtrip(he_mf):
    occ_idx = numpy.array([0], dtype=int)
    e = numpy.array([0.1, 0.2])
    xy = (
        (numpy.zeros((1, 3)), numpy.zeros((1, 3))),
        (numpy.zeros((1, 3)), numpy.zeros((1, 3))),
    )

    manifold = Manifold(
        mol=he_mf.mol,
        mo_coeff=he_mf.mo_coeff,
        mo_occ=he_mf.mo_occ,
        occ_idx=occ_idx,
        e=e,
        xy=xy,
    )
    s = manifold.dump()
    assert isinstance(s, str)
    payload = json.loads(s)
    assert payload['xy'][0][1] is not None
    assert 'frozen_idx' not in payload
    assert 'mol' not in payload
    assert 'mo_coeff' not in payload

    restored = Manifold.loads(s, he_mf)
    numpy.testing.assert_array_equal(restored.occ_idx, occ_idx)
    numpy.testing.assert_array_equal(restored.e, e)
    assert len(restored.xy) == len(xy)
    for (x1, y1), (x2, y2) in zip(restored.xy, xy):
        numpy.testing.assert_array_equal(x1, x2)
        if y1 is None:
            assert y2 is None
        else:
            numpy.testing.assert_array_equal(y1, y2)
    assert restored.mol is he_mf.mol
    assert restored.mo_coeff is he_mf.mo_coeff


def test_manifold_from_tdobj(he_mf):
    td = RPA(he_mf).set(nstates=1)
    td.kernel()
    manifold = Manifold.from_tdobj(td)
    assert manifold.e.shape == (1,)
    assert len(manifold.xy) == 1
    assert manifold.xy[0][1] is not None
    assert len(manifold.occ_idx) == 1
    assert manifold.mol is he_mf.mol
    assert manifold.mo_coeff is he_mf.mo_coeff


def test_manifold_from_tda_tdobj(he_mf):
    td = TDA(he_mf).set(nstates=1)
    td.kernel()
    manifold = Manifold.from_tdobj(td)
    assert manifold.xy[0][1] is None


