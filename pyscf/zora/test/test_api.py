'''API tests for the ZORA mixin, helper options, and scanners.'''

import io
import numpy as np
import pytest
import pyscf
import pyscf.zora
from pyscf.zora import integrals
from pyscf.zora.test.common import GRAD_GRID_LEVEL, make_hf_mol, make_rhf


def test_spin_orbit_ignored_on_rhf():
    mol = make_hf_mol()
    mf = pyscf.scf.RHF(mol).zora(spin_orbit=True, grid_level=GRAD_GRID_LEVEL)
    assert mf.with_zora.spin_orbit is False
    assert np.isrealobj(mf.get_hcore())


def test_custom_grid_matches_grid_level():
    mol = make_hf_mol()
    grid = integrals.build_zora_grid(mol, level=GRAD_GRID_LEVEL)
    h_grid = pyscf.scf.RHF(mol).zora(grid=grid).get_hcore()
    h_level = pyscf.scf.RHF(mol).zora(grid_level=GRAD_GRID_LEVEL).get_hcore()
    assert np.allclose(h_grid, h_level)


def test_hcore_cache_and_get_hso():
    mol = make_hf_mol()
    mf = make_rhf(mol)
    h1 = mf.get_hcore()
    h2 = mf.get_hcore()
    assert h1 is h2
    assert mf.with_zora.get_hso() is None


def test_hessian_not_implemented():
    mol = make_hf_mol()
    mf = make_rhf(mol)
    with pytest.raises(NotImplementedError):
        mf.with_zora.hcore_deriv_generator(deriv=2)


def test_dump_flags():
    mol = make_hf_mol()
    mf = make_rhf(mol)
    buf = io.StringIO()
    mf.stdout = buf
    mf.with_zora.stdout = buf
    mf.verbose = 4
    mf.with_zora.verbose = 4
    mf.dump_flags()
    text = buf.getvalue()
    assert 'grid_level' in text
    assert 'spin_orbit' in text


def test_as_scanner_rebuilds_hcore():
    mol = make_hf_mol()
    mf = make_rhf(mol)
    mf.kernel()
    scanner = mf.as_scanner()
    e1 = scanner(mol)
    mol2 = mol.copy()
    coords = mol.atom_coords().copy()
    coords[1, 2] += 0.3
    mol2.set_geom_(coords, unit='Bohr')
    e2 = scanner(mol2)
    assert abs(e1 - e2) > 1e-4


def test_to_ks_keeps_zora():
    mol = make_hf_mol()
    mf = make_rhf(mol)
    mf.kernel()
    ks = mf.to_ks('lda,vwn')
    assert hasattr(ks, 'with_zora')
    assert ks.with_zora.grid_level == GRAD_GRID_LEVEL


def test_gradients_alias():
    mol = make_hf_mol()
    mf = make_rhf(mol)
    mf.kernel()
    g1 = mf.nuc_grad_method()
    g2 = mf.Gradients()
    assert np.allclose(g1.kernel(), g2.kernel())


def test_generalized_gradients_require_pyscf_support(monkeypatch):
    from pyscf.zora import grad as zora_grad

    monkeypatch.setattr(zora_grad, '_ghf_grad', None)
    mf = pyscf.scf.GHF(make_hf_mol()).zora(grid_level=GRAD_GRID_LEVEL)
    with pytest.raises(NotImplementedError, match='generalized'):
        mf.nuc_grad_method()


def test_idempotent_updates_options():
    mol = make_hf_mol()
    mf = pyscf.scf.RHF(mol).zora(grid_level=8)
    h8 = mf.get_hcore().copy()
    mf = mf.zora(grid_level=GRAD_GRID_LEVEL)
    h3 = mf.get_hcore()
    assert mf.with_zora.grid_level == GRAD_GRID_LEVEL
    assert not np.allclose(h8, h3)
