'''Frozen-grid finite-difference checks for MP-ZORA integral derivatives.'''

import numpy as np
from pyscf.zora import integrals
from pyscf.zora.test.common import (
    FD_AO_TOL, GRAD_GRID_LEVEL, full_T_grad, Hso_frozen, make_hf_mol,
)


def test_dveff_frozen_grid():
    mol = make_hf_mol(1.1)
    grid = integrals.build_zora_grid(mol, level=GRAD_GRID_LEVEL)
    ia = 1
    c, a = integrals.modbas_for_mol(mol)[ia]
    Z = mol.atom_charge(ia)
    xyz = mol.atom_coord(ia)
    dveff = integrals.eval_dveff_dRA(grid.coords, xyz, c, a, Z)

    R = np.sqrt(np.sum((grid.coords - xyz)**2, axis=1))
    mask = R > 0.1
    mol1 = mol.copy()
    coords = mol.atom_coords()
    delta = 1e-5
    errmax = 0.0
    for alpha in range(3):
        coords[ia, alpha] += delta
        mol1.set_geom_(coords, unit='Bohr')
        vp = integrals.eval_model_potential(mol1, grid.coords)
        coords[ia, alpha] -= 2 * delta
        mol1.set_geom_(coords, unit='Bohr')
        vm = integrals.eval_model_potential(mol1, grid.coords)
        coords[ia, alpha] += delta
        fd = (vp - vm) / (2 * delta)
        errmax = max(errmax, np.max(np.abs(fd[mask] - dveff[alpha, mask])))
    assert errmax < FD_AO_TOL, errmax


def test_T_ao_grad_frozen_grid():
    mol = make_hf_mol(1.1)
    ia = 1
    delta = 1e-5
    grid = integrals.build_zora_grid(mol, level=GRAD_GRID_LEVEL)
    kernel0 = integrals.zora_kernel(integrals.eval_model_potential(mol, grid.coords))
    dT = full_T_grad(mol, grid, kernel0, ia)

    def T_frozen(m):
        kern = integrals.zora_kernel(integrals.eval_model_potential(m, grid.coords))
        T, _ = integrals.eval_zora_T(m, grid, kern)
        return T

    mol1 = mol.copy()
    coords = mol.atom_coords()
    errmax = 0.0
    for alpha in range(3):
        coords[ia, alpha] += delta
        mol1.set_geom_(coords, unit='Bohr')
        Tp = T_frozen(mol1)
        coords[ia, alpha] -= 2 * delta
        mol1.set_geom_(coords, unit='Bohr')
        Tm = T_frozen(mol1)
        coords[ia, alpha] += delta
        fd = (Tp - Tm) / (2 * delta)
        errmax = max(errmax, np.max(np.abs(fd - dT[alpha])))
    assert errmax < FD_AO_TOL, errmax


def test_Hso_ao_grad_frozen_grid():
    mol = make_hf_mol(1.1)
    ia = 1
    delta = 1e-5
    grid = integrals.build_zora_grid(mol, level=GRAD_GRID_LEVEL)
    veff0 = integrals.eval_model_potential(mol, grid.coords)
    kernel0 = integrals.zora_kernel(veff0)
    dHso = integrals.eval_zora_SO_grad(mol, grid, kernel0, veff0)

    mol1 = mol.copy()
    coords = mol.atom_coords()
    errmax = 0.0
    for alpha in range(3):
        coords[ia, alpha] += delta
        mol1.set_geom_(coords, unit='Bohr')
        Hp = Hso_frozen(mol1, grid)
        coords[ia, alpha] -= 2 * delta
        mol1.set_geom_(coords, unit='Bohr')
        Hm = Hso_frozen(mol1, grid)
        coords[ia, alpha] += delta
        fd = (Hp - Hm) / (2 * delta)
        errmax = max(errmax, np.max(np.abs(fd - dHso[ia, alpha])))
    assert errmax < FD_AO_TOL, errmax
