'''Focused edge-case tests for QR coverage.'''

from types import SimpleNamespace

import numpy
import pytest
from pyscf import dft, gto, scf
from pyscf.tdscf import TDA

from pyscf.qr import Manifold, QR
from pyscf.qr import chkfile as qr_chkfile
from pyscf.qr.ghf import GQR
from pyscf.qr.hf import QR as BaseQR
from pyscf.qr.hf import qr_class_for_mf
from pyscf.qr.manifold import (
    _active_occ_idx,
    _decode_xy,
    _encode_xy,
    check_shared_reference,
)
from pyscf.qr.rhf import (
    Gxc,
    _compute_c,
    _get_pq,
    _precompute_gxc,
)
from pyscf.qr.uhf import UQR


def _mf_data():
    mol = object()
    mo_coeff = numpy.eye(3)
    mo_occ = numpy.array([2, 2, 0])
    return SimpleNamespace(mol=mol, mo_coeff=mo_coeff, mo_occ=mo_occ)


def _manifold(mf=None, *, y=0, nvirt=1):
    if mf is None:
        mf = _mf_data()
    return Manifold(
        mol=mf.mol,
        mo_coeff=mf.mo_coeff,
        mo_occ=mf.mo_occ,
        occ_idx=numpy.array([0]),
        e=numpy.array([0.2]),
        xy=((numpy.ones((1, nvirt)), y),),
    )


def test_checkpoint_save_noop_and_secondary_manifold(monkeypatch):
    primary = _manifold()
    secondary = _manifold()
    qrobj = SimpleNamespace(
        chkfile=None,
        manifold_n=primary,
        manifold_m=secondary,
    )
    qr_chkfile.save_qr(qrobj)

    saved = []
    monkeypatch.setattr(
        qr_chkfile.lib.chkfile,
        'save',
        lambda path, key, value: saved.append((path, key, value)),
    )
    qrobj.chkfile = 'qr.chk'
    qr_chkfile.save_qr(qrobj)
    assert [key for _, key, _ in saved] == [
        'qr/manifold_n',
        'qr/manifold_m',
    ]


def test_checkpoint_secondary_fallbacks(monkeypatch):
    mf = _mf_data()
    primary = _manifold(mf)
    calls = []

    def missing_secondary(path, key):
        calls.append((path, key))
        raise KeyError(key)

    monkeypatch.setattr(qr_chkfile.lib.chkfile, 'load', missing_secondary)
    assert qr_chkfile.load_manifold_m('qr.chk', mf, primary) is primary
    assert calls == [('qr.chk', 'qr/manifold_m')]

    monkeypatch.setattr(
        qr_chkfile.lib.chkfile,
        'load',
        lambda path, key: None,
    )
    assert qr_chkfile.load_manifold_m('qr.chk', mf, primary) is primary

    payload = primary.dump()
    monkeypatch.setattr(
        qr_chkfile.lib.chkfile,
        'load',
        lambda path, key: payload,
    )
    restored = qr_chkfile.load_manifold_m('qr.chk', mf, primary)
    numpy.testing.assert_array_equal(restored.e, primary.e)


def test_reference_dispatch_and_unimplemented_methods():
    mol = gto.M(atom='H 0 0 0', basis='sto-3g', spin=1, verbose=0)
    assert qr_class_for_mf(scf.GHF(mol)).__name__ == 'GQR'
    assert qr_class_for_mf(scf.UHF(mol)).__name__ == 'UQR'
    assert qr_class_for_mf(scf.RHF(mol)).__name__ == 'RQR'
    with pytest.raises(TypeError, match='Unsupported mean-field type'):
        qr_class_for_mf(object())

    with pytest.raises(NotImplementedError):
        GQR.kernel(GQR.__new__(GQR))
    with pytest.raises(NotImplementedError):
        GQR.get_2tdm(GQR.__new__(GQR), 0, 0)
    with pytest.raises(NotImplementedError):
        UQR.kernel(UQR.__new__(UQR))
    with pytest.raises(NotImplementedError):
        UQR.get_2tdm(UQR.__new__(UQR), 0, 0)


def test_base_qr_error_paths():
    base = BaseQR.__new__(BaseQR)
    with pytest.raises(ValueError, match='Could not infer'):
        BaseQR._infer_response_type(
            SimpleNamespace(xy=((numpy.zeros((1, 1)), object()),)),
        )
    for y in (0, numpy.int64(0), numpy.asarray(0), (0, 0)):
        assert BaseQR._infer_response_type(
            SimpleNamespace(xy=((numpy.zeros((1, 1)), y),))) == 'tda'
    assert BaseQR._infer_response_type(
        SimpleNamespace(xy=((numpy.zeros((1, 1)),
                             numpy.zeros((1, 1))),))) == 'rpa'
    with pytest.raises(NotImplementedError, match='_init_gxc'):
        base._init_gxc()
    with pytest.raises(NotImplementedError, match='_build_intermediates'):
        base._build_intermediates()

    manifold = _manifold()
    driver = SimpleNamespace(_manifold_n=manifold, _manifold_m=manifold)
    with pytest.raises(IndexError, match='i=-1'):
        BaseQR._sanity_check_2tdm(driver, -1, 0)
    with pytest.raises(IndexError, match='j=1'):
        BaseQR._sanity_check_2tdm(driver, 0, 1)


def test_manifold_edge_cases():
    mf = _mf_data()
    numpy.testing.assert_array_equal(_active_occ_idx(mf, 1), numpy.array([1]))
    numpy.testing.assert_array_equal(_active_occ_idx(mf, [1]), numpy.array([0]))
    with pytest.raises(NotImplementedError, match='frozen'):
        _active_occ_idx(mf, 1.5)

    decoded = _decode_xy([
        [[1.0], 0],
        [[2.0], [0, 0]],
        [[3.0], [1.0, 2.0]],
    ])
    assert decoded[0][1] == 0
    assert decoded[1][1] == (0, 0)
    numpy.testing.assert_array_equal(decoded[2][1], numpy.array([1.0, 2.0]))

    x = numpy.ones((1, 1))
    encoded = _encode_xy(((x, 0), (x, (0, 0))))
    assert encoded[0][1] == encoded[1][1] == 0
    assert encoded[0][0] == encoded[1][0] == x.tolist()
    with pytest.raises(ValueError, match='Unknown type'):
        _encode_xy(((x, object()),))

    with pytest.raises(ValueError, match='len'):
        Manifold(
            mol=mf.mol,
            mo_coeff=mf.mo_coeff,
            mo_occ=mf.mo_occ,
            occ_idx=[0],
            e=[0.1],
            xy=((x, 0), (x, 0)),
        )
    with pytest.raises(ValueError, match='TDA or RPA'):
        Manifold(
            mol=mf.mol,
            mo_coeff=mf.mo_coeff,
            mo_occ=mf.mo_occ,
            occ_idx=[0],
            e=[0.1, 0.2],
            xy=((x, 0), (x, numpy.ones((1, 1)))),
        )

    man = _manifold(mf, nvirt=1)
    with pytest.raises(IndexError, match='state=-1'):
        man.get_aligned_xy(-1)
    with pytest.raises(IndexError, match='state=1'):
        man(1)
    too_wide = Manifold(
        mol=mf.mol,
        mo_coeff=mf.mo_coeff,
        mo_occ=mf.mo_occ,
        occ_idx=[0],
        e=[0.1],
        xy=((numpy.ones((1, 2)), 0),),
    )
    with pytest.raises(ValueError, match='virtual columns'):
        too_wide.get_aligned_xy(0)
    assert man.dumps() == man.dump()


def test_shared_reference_validation():
    mol = object()
    coeff = numpy.eye(2)
    td1 = SimpleNamespace(_scf=SimpleNamespace(mol=mol, mo_coeff=coeff))
    td2 = SimpleNamespace(_scf=SimpleNamespace(mol=object(), mo_coeff=coeff))
    with pytest.raises(ValueError, match='same mol'):
        check_shared_reference(td1, td2)

    td2._scf.mol = mol
    td2._scf.mo_coeff = numpy.array(coeff)
    with pytest.raises(ValueError, match='same mo_coeff'):
        check_shared_reference(td1, td2)


def test_tda_pq_branch():
    c = numpy.arange(4.0).reshape(1, 1, 2, 2)
    x1 = numpy.array([[0.3]])
    x2 = numpy.array([[0.4]])
    knm = numpy.zeros((2, 2))
    v = numpy.zeros((1, 1))
    pia, qia = _get_pq(c, knm, v, x1, x2, 0, 0)
    assert pia.shape == qia.shape == (1, 1)


@pytest.fixture
def h2_rks():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    return dft.RKS(mol, xc='LDA').run()


def test_unsupported_xc_branches(h2_rks, monkeypatch):
    mf = h2_rks
    monkeypatch.setattr(mf._numint, '_xc_type', lambda xc: 'MGGA')
    nocc = int(numpy.count_nonzero(mf.mo_occ == 2))
    nvirt = int(numpy.count_nonzero(mf.mo_occ == 0))
    shape = (nocc, nvirt, nocc, nvirt, nocc, nvirt)
    with pytest.raises(NotImplementedError, match='xctype = MGGA'):
        _precompute_gxc(mf, numpy.zeros(shape), [0], [0])
    with pytest.raises(NotImplementedError, match='xctype = MGGA'):
        _compute_c(mf)

    manifold = _manifold(
        SimpleNamespace(
            mol=mf.mol,
            mo_coeff=mf.mo_coeff,
            mo_occ=mf.mo_occ,
        ),
    )
    backend = Gxc(
        SimpleNamespace(
            _manifold_n=manifold,
            _manifold_m=manifold,
            mo_occ=mf.mo_occ,
        ),
    )
    with pytest.raises(NotImplementedError, match='xctype = MGGA'):
        backend.contract_v(mf, numpy.ones((nocc, nvirt)),
                           numpy.ones((nocc, nvirt)))


def test_rsh_compute_c(h2_rks, monkeypatch):
    mf = h2_rks
    monkeypatch.setattr(
        mf._numint,
        'rsh_and_hybrid_coeff',
        lambda xc, spin: (0.5, 0.2, 0.1),
    )
    c = _compute_c(mf)
    assert c.shape[:2] == (1, 1)


def test_gxc_precompute_lazy_noop_and_invalid_approximation():
    mf = _mf_data()
    man = _manifold(mf)
    backend = Gxc(
        SimpleNamespace(_manifold_n=man, _manifold_m=man, mo_occ=mf.mo_occ),
        precompute_gxc=False,
    )
    assert backend.precompute(mf) is backend

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    rhf = scf.RHF(mol).run()
    td = TDA(rhf).set(nstates=1)
    td.kernel()
    with pytest.raises(ValueError, match='Invalid approxiamtion'):
        QR(td, approximation='invalid').get_2tdm(0, 0)
