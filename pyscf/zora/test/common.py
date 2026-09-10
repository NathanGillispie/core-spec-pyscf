'''Shared molecules and helpers for MP-ZORA nuclear-gradient tests.'''

import numpy as np
import pytest
import pyscf
import pyscf.zora
from pyscf.zora import integrals

FD_AO_TOL = 1e-5
ENERGY_GRAD_TOL = 5e-4
GEOMOPT_GNORM_TOL = 5e-4
GRAD_GRID_LEVEL = 3

CONV_PARAMS = {
    'convergence_energy': 1e-6,
    'convergence_grms': 3e-4,
    'convergence_gmax': 4.5e-4,
    'convergence_drms': 1.2e-3,
    'convergence_dmax': 1.8e-3,
}

try:
    from pyscf.grad import ghf as _ghf_grad  # noqa: F401
    from pyscf.grad import gks as _gks_grad  # noqa: F401
except ImportError:
    _ghf_grad = None
    _gks_grad = None

requires_ghf_grad = pytest.mark.skipif(
    _ghf_grad is None or _gks_grad is None,
    reason='GHF/GKS nuclear gradients require PySCF generalized '
           'nuclear-gradient support',
)


def make_hf_mol(bond_ang=1.1, verbose=0):
    return pyscf.gto.M(
        atom=f'H 0 0 0; F 0 0 {bond_ang}',
        basis='sto-3g',
        unit='Angstrom',
        verbose=verbose,
        symmetry=False,
    )


def _zora(mf):
    return mf.zora(grid_level=GRAD_GRID_LEVEL)


def make_rhf(mol, conv_tol=1e-12):
    mf = pyscf.scf.RHF(mol)
    mf.verbose = 0
    mf.conv_tol = conv_tol
    return _zora(mf)


def make_rks(mol, conv_tol=1e-10):
    mf = pyscf.dft.RKS(mol)
    mf.xc = 'lda,vwn'
    mf.verbose = 0
    mf.conv_tol = conv_tol
    return _zora(mf)


def make_uhf(mol, conv_tol=1e-12):
    mf = pyscf.scf.UHF(mol)
    mf.verbose = 0
    mf.conv_tol = conv_tol
    return _zora(mf)


def make_uks(mol, conv_tol=1e-10):
    mf = pyscf.dft.UKS(mol)
    mf.xc = 'lda,vwn'
    mf.verbose = 0
    mf.conv_tol = conv_tol
    return _zora(mf)


def make_ghf(mol, conv_tol=1e-12, spin_orbit=False):
    mf = pyscf.scf.GHF(mol)
    mf.verbose = 0
    mf.conv_tol = conv_tol
    return mf.zora(spin_orbit=spin_orbit, grid_level=GRAD_GRID_LEVEL)


def make_gks(mol, conv_tol=1e-10, spin_orbit=False):
    mf = pyscf.dft.GKS(mol)
    mf.xc = 'lda,vwn'
    mf.collinear = 'col'
    mf.verbose = 0
    mf.conv_tol = conv_tol
    return mf.zora(spin_orbit=spin_orbit, grid_level=GRAD_GRID_LEVEL)


def make_rohf(mol, conv_tol=1e-12):
    mf = pyscf.scf.ROHF(mol)
    mf.verbose = 0
    mf.conv_tol = conv_tol
    return _zora(mf)


def energy_fd_error(make_mf, mol, delta=1e-4):
    '''max |analytic − FD| of the SCF energy nuclear gradient.'''
    mf = make_mf(mol)
    mf.kernel()
    g = mf.nuc_grad_method()
    g.verbose = 0
    de = g.kernel()

    def energy(coords):
        m = mol.copy()
        m.set_geom_(coords, unit='Bohr')
        return make_mf(m).kernel()

    coords = mol.atom_coords()
    de_fd = np.zeros_like(de)
    for ia in range(mol.natm):
        for alpha in range(3):
            coords[ia, alpha] += delta
            ep = energy(coords)
            coords[ia, alpha] -= 2 * delta
            em = energy(coords)
            coords[ia, alpha] += delta
            de_fd[ia, alpha] = (ep - em) / (2 * delta)
    return np.max(np.abs(de - de_fd)), de, de_fd


def geomopt_gnorm(make_mf, mol):
    '''Optimize, then return ||∇E|| and the equilibrium distance (Bohr).'''
    from pyscf.geomopt import geometric_solver
    mf = make_mf(mol)
    mol_eq = geometric_solver.optimize(mf, **CONV_PARAMS)
    mf_eq = make_mf(mol_eq)
    mf_eq.kernel()
    de = mf_eq.nuc_grad_method().kernel()
    r = np.linalg.norm(mol_eq.atom_coords()[1] - mol_eq.atom_coords()[0])
    return np.linalg.norm(de), r, de


def full_T_grad(mol, grid, kernel, ia):
    T, ipkin = integrals.eval_zora_T(mol, grid, kernel, deriv_bra=True)
    dT_kern = integrals.eval_zora_T_kernel_deriv(mol, grid, kernel)
    p0, p1 = mol.aoslice_by_atom()[ia, 2:]
    pulay = np.zeros_like(ipkin)
    pulay[:, p0:p1] = -ipkin[:, p0:p1]
    pulay = pulay + pulay.transpose(0, 2, 1)
    return pulay + dT_kern[ia]


def Hso_frozen(mol, grid):
    veff = integrals.eval_model_potential(mol, grid.coords)
    kernel = integrals.zora_kernel(veff)
    Hx, Hy, Hz = integrals.eval_zora_SO(mol, grid, kernel * veff / 2.)
    return integrals.assemble_Hso(Hx, Hy, Hz)
