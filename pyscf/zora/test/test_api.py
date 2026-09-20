'''API tests for the ZORA mixin, helper options, and scanners.'''

import io
import numpy as np
import pytest
import pyscf
import pyscf.zora
from pyscf.tdscf import TDA
from pyscf.zora import integrals
from pyscf.zora.test.common import GRAD_GRID_LEVEL, make_hf_mol, make_rhf, make_rks


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


def test_mo_energy_correction_is_default_and_can_be_disabled():
    mol = make_hf_mol()
    mf = pyscf.scf.RHF(mol).zora(grid_level=GRAD_GRID_LEVEL)
    mf_no_corr = pyscf.scf.RHF(mol).zora(
        grid_level=GRAD_GRID_LEVEL, mo_energy_correction=False)
    mf.verbose = mf_no_corr.verbose = 0
    mf.kernel()
    mf_no_corr.kernel()

    assert mf.with_zora.mo_energy_correction is True
    assert mf_no_corr.with_zora.mo_energy_correction is False

    occ_idx = np.where(mf.mo_occ != 0)[0]
    eps_scal_mo = (mf.mo_coeff[:, occ_idx].conj().T
                   @ mf.with_zora._eps_scal_ao @ mf.mo_coeff[:, occ_idx])
    expected = mf_no_corr.mo_energy.copy()
    expected[occ_idx] *= (1 + np.diag(eps_scal_mo).real)**-1
    assert np.allclose(mf.mo_energy, expected)
    assert np.allclose(mf.mo_energy[mf.mo_occ == 0],
                       mf_no_corr.mo_energy[mf.mo_occ == 0])


def test_uhf_mo_energy_correction_changes_occupied_energies():
    mol = make_hf_mol()
    mf = pyscf.scf.UHF(mol).zora(grid_level=GRAD_GRID_LEVEL)
    mf_no_corr = pyscf.scf.UHF(mol).zora(
        grid_level=GRAD_GRID_LEVEL, mo_energy_correction=False)
    mf.verbose = mf_no_corr.verbose = 0
    mf.kernel()
    mf_no_corr.kernel()

    for spin in range(2):
        occ_idx = mf.mo_occ[spin] != 0
        virt_idx = ~occ_idx
        assert not np.allclose(mf.mo_energy[spin][occ_idx],
                               mf_no_corr.mo_energy[spin][occ_idx])
        assert np.allclose(mf.mo_energy[spin][virt_idx],
                           mf_no_corr.mo_energy[spin][virt_idx])


def test_ghf_mo_energy_correction_changes_occupied_energies():
    mol = make_hf_mol()
    mf = pyscf.scf.GHF(mol).zora(grid_level=GRAD_GRID_LEVEL)
    mf_no_corr = pyscf.scf.GHF(mol).zora(
        grid_level=GRAD_GRID_LEVEL, mo_energy_correction=False)
    mf.verbose = mf_no_corr.verbose = 0
    mf.kernel()
    mf_no_corr.kernel()

    occ_idx = mf.mo_occ != 0
    virt_idx = ~occ_idx
    assert not np.allclose(mf.mo_energy[occ_idx],
                           mf_no_corr.mo_energy[occ_idx])
    assert np.allclose(mf.mo_energy[virt_idx],
                       mf_no_corr.mo_energy[virt_idx])


def test_zora_tda_uses_corrected_mo_energy():
    mol = make_hf_mol()
    mf = make_rks(mol)
    mf_no_corr = pyscf.dft.RKS(mol, xc='lda,vwn').zora(
        grid_level=GRAD_GRID_LEVEL, mo_energy_correction=False)
    mf.verbose = mf_no_corr.verbose = 0
    mf.kernel()
    mf_no_corr.kernel()

    td = TDA(mf).set(nstates=1)
    td_no_corr = TDA(mf_no_corr).set(nstates=1)
    td.verbose = td_no_corr.verbose = 0
    td.kernel()
    td_no_corr.kernel()

    a, _ = td.get_ab()
    a_no_corr, _ = td_no_corr.get_ab()
    assert not np.allclose(a, a_no_corr)
    assert not np.allclose(td.e, td_no_corr.e)
