'''MP-ZORA analytic nuclear gradients vs finite difference.'''

import numpy as np
from pyscf.zora.test.common import (
    ENERGY_GRAD_TOL, energy_fd_error, make_ghf, make_hf_mol, make_rhf, make_uhf,
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


def test_ghf_so_energy_grad():
    err, _, _ = energy_fd_error(
        lambda m: make_ghf(m, spin_orbit=True), make_hf_mol(1.1))
    assert err < ENERGY_GRAD_TOL, err
