import numpy
import pytest

from pyscf import gto, lib, scf
from pyscf.tdscf import TDA

import pyscf.qr
import pyscf.zora
from pyscf.qr import QR
from pyscf.rixs import RIXS
from pyscf.zora.zora import ZORA_SCF


def _make_mf(zora=False):
    mol = gto.M(
        atom='H 0 0 0; H 0 0 0.74',
        basis='sto-3g',
        verbose=0,
    )
    mf = scf.RHF(mol)
    if zora:
        mf = mf.zora(grid_level=1)
    return mf.run()


def _make_rixs(mf, chkfile=None):
    td = TDA(mf).set(nstates=1)
    qr = QR(td)
    return RIXS(mf, qr, chkfile=chkfile)


def test_public_api_and_reference_validation():
    mf = _make_mf()
    rixs = _make_rixs(mf)

    assert rixs.mf is mf
    assert rixs.qr.mf is mf
    assert rixs.mol is mf.mol
    assert rixs.chkfile == rixs.qr.chkfile

    with pytest.raises(TypeError, match='QR driver'):
        RIXS(mf, object())
    with pytest.raises(ValueError, match='same mean-field'):
        RIXS(_make_mf(), rixs.qr)


def test_ground_transition_dipoles():
    mf = _make_mf()
    td = TDA(mf).set(nstates=1)
    td.kernel()
    rixs = RIXS(mf, QR(td))

    state = 0
    actual = rixs.ground_transition_dipoles([state])

    expected = td.transition_dipole()[[state]].T

    numpy.testing.assert_allclose(actual, expected)


def test_checkpoint_roundtrip(tmp_path):
    chkfile = str(tmp_path / 'rixs.chk')
    rixs = _make_rixs(_make_mf(), chkfile=chkfile)
    rixs.save()

    with lib.H5FileWrap(chkfile, 'r') as fh5:
        assert 'mol' in fh5
        assert 'scf' in fh5
        assert 'qr/manifold_n' in fh5
        assert 'qr/dipole_mo' in fh5
        assert 'rixs/zora' in fh5

    template = _make_mf()
    restored = RIXS.from_chk(chkfile, template)
    assert restored.mf is template
    assert type(restored.qr).__name__ == 'RQR'
    numpy.testing.assert_array_equal(
        restored.qr.manifold_n.e,
        rixs.qr.manifold_n.e,
    )
    numpy.testing.assert_allclose(restored.dipole_mo, rixs.dipole_mo)


def test_save_path_override_and_noop(tmp_path):
    rixs = _make_rixs(_make_mf())
    assert rixs.save() is rixs
    chkfile = str(tmp_path / 'override.chk')
    assert rixs.save(chkfile) is rixs
    with lib.H5FileWrap(chkfile, 'r') as fh5:
        assert 'qr/manifold_n' in fh5


def test_checkpoint_roundtrip_reapplies_zora(tmp_path):
    chkfile = str(tmp_path / 'rixs-zora.chk')
    rixs = _make_rixs(_make_mf(zora=True), chkfile=chkfile)
    rixs.save()

    restored = RIXS.from_chk(chkfile, _make_mf())
    assert isinstance(restored.mf, ZORA_SCF)
    assert restored.mf.with_zora.grid_level == 1
    assert restored.mf.with_zora.spin_orbit is False
    numpy.testing.assert_allclose(
        restored.mf.mo_coeff,
        rixs.mf.mo_coeff,
    )
