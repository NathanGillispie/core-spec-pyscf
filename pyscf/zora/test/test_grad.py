'''MP-ZORA analytic nuclear gradients vs finite difference.'''

import numpy as np
import pyscf
import pyscf.zora
from pyscf.zora.grad import make_grad_object
from pyscf.zora.test.common import (
    ENERGY_GRAD_TOL,
    GRAD_GRID_LEVEL,
    energy_fd_error,
    make_ghf,
    make_gks,
    make_hf_mol,
    make_rhf,
    make_rks,
    make_uhf,
    make_uks,
    requires_ghf_grad,
)


def test_rhf_energy_grad():
    err, _, _ = energy_fd_error(make_rhf, make_hf_mol(1.1))
    assert err < ENERGY_GRAD_TOL, err


def test_uhf_matches_rhf_closed_shell():
    mol = make_hf_mol(1.1)
    mf_r = make_rhf(mol)
    mf_r.kernel()
    de_r = mf_r.nuc_grad_method().kernel()
    mf_u = make_uhf(mol)
    mf_u.kernel()
    de_u = mf_u.nuc_grad_method().kernel()
    assert np.max(np.abs(de_r - de_u)) < 1e-8
    assert abs(mf_r.e_tot - mf_u.e_tot) < 1e-8


def test_uhf_energy_grad():
    err, _, _ = energy_fd_error(make_uhf, make_hf_mol(1.1))
    assert err < ENERGY_GRAD_TOL, err


@requires_ghf_grad
def test_ghf_matches_rhf_scalar():
    mol = make_hf_mol(1.1)
    mf_r = make_rhf(mol)
    mf_r.kernel()
    de_r = mf_r.nuc_grad_method().kernel()
    mf_g = make_ghf(mol)
    mf_g.kernel()
    de_g = mf_g.nuc_grad_method().kernel()
    assert np.max(np.abs(de_r - de_g)) < 1e-8
    assert abs(mf_r.e_tot - mf_g.e_tot) < 1e-8


@requires_ghf_grad
def test_ghf_so_energy_grad():
    err, _, _ = energy_fd_error(lambda m: make_ghf(m, spin_orbit=True),
                                make_hf_mol(1.1))
    assert err < ENERGY_GRAD_TOL, err


def test_rks_energy_grad():
    err, _, _ = energy_fd_error(make_rks, make_hf_mol(1.1))
    assert err < ENERGY_GRAD_TOL, err


def test_uks_matches_rks_closed_shell():
    mol = make_hf_mol(1.1)
    mf_r = make_rks(mol)
    mf_r.kernel()
    de_r = mf_r.nuc_grad_method().kernel()
    mf_u = make_uks(mol)
    mf_u.kernel()
    de_u = mf_u.nuc_grad_method().kernel()
    assert np.max(np.abs(de_r - de_u)) < 1e-6
    assert abs(mf_r.e_tot - mf_u.e_tot) < 1e-7


@requires_ghf_grad
def test_gks_matches_rks_scalar():
    mol = make_hf_mol(1.1)
    mf_r = make_rks(mol)
    mf_r.kernel()
    de_r = mf_r.nuc_grad_method().kernel()
    mf_g = make_gks(mol)
    mf_g.kernel()
    de_g = mf_g.nuc_grad_method().kernel()
    assert np.max(np.abs(de_r - de_g)) < 1e-6


@requires_ghf_grad
def test_gks_so_energy_grad():
    err, _, _ = energy_fd_error(lambda m: make_gks(m, spin_orbit=True),
                                make_hf_mol(1.1))
    assert err < ENERGY_GRAD_TOL, err


def test_make_grad_object_from_gradients():
    mol = make_hf_mol()
    mf = make_rhf(mol)
    mf.kernel()
    g0 = mf.Gradients()
    de0 = g0.kernel()
    g1 = make_grad_object(g0)
    de1 = g1.kernel()
    assert np.allclose(de0, de1)


def test_density_fit_then_zora_grad():
    mol = make_hf_mol()
    mf = pyscf.scf.RHF(mol).density_fit().zora(grid_level=GRAD_GRID_LEVEL)
    mf.verbose = 0
    mf.conv_tol = 1e-10
    mf.kernel()
    de = mf.Gradients().kernel()
    assert de.shape == (mol.natm, 3)
    assert np.linalg.norm(de) > 0


def test_ecp_energy_and_grad():
    mol = pyscf.gto.M(
        atom='Cl 0 0 0; H 0 0 1.3',
        basis='lanl2dz',
        ecp='lanl2dz',
        verbose=0,
        unit='Angstrom',
    )
    mf = pyscf.scf.RHF(mol).zora(grid_level=GRAD_GRID_LEVEL)
    mf.verbose = 0
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mol.has_ecp()
    de = mf.Gradients().kernel()
    assert de.shape == (2, 3)
    assert np.linalg.norm(de) > 0
