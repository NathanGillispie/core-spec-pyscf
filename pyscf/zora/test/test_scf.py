'''Energy coverage for HF/KS flavors that the original plugin attached but did not test.'''

import numpy as np
import pyscf
import pyscf.zora
from pyscf.zora.test.common import (
    GRAD_GRID_LEVEL, make_ghf, make_gks, make_hf_mol, make_rhf, make_rks,
    make_rohf, make_uhf, make_uks,
)


def test_uhf_closed_shell_energy():
    mol = make_hf_mol()
    mf_r = make_rhf(mol)
    mf_u = make_uhf(mol)
    assert abs(mf_r.kernel() - mf_u.kernel()) < 1e-8


def test_ghf_scalar_energy():
    mol = make_hf_mol()
    e_r = make_rhf(mol).kernel()
    e_g = make_ghf(mol).kernel()
    assert abs(e_r - e_g) < 1e-8


def test_ghf_so_energy_differs_and_hcore_hermitian():
    mol = make_hf_mol()
    e_sf = make_ghf(mol).kernel()
    mf_so = make_ghf(mol, spin_orbit=True)
    e_so = mf_so.kernel()
    h = mf_so.get_hcore()
    assert abs(e_so - e_sf) > 1e-6
    assert np.allclose(h, h.conj().T)
    hso = mf_so.with_zora.get_hso()
    assert hso is not None
    assert np.allclose(1j * hso, (1j * hso).conj().T)


def test_rks_energy_differs_from_nr():
    mol = make_hf_mol()
    mf = pyscf.dft.RKS(mol)
    mf.xc = 'lda,vwn'
    mf.verbose = 0
    e_nr = mf.kernel()
    mf = mf.zora(grid_level=GRAD_GRID_LEVEL)
    e_z = mf.kernel()
    assert abs(e_z - e_nr) > 1e-4


def test_uks_closed_shell_energy():
    mol = make_hf_mol()
    assert abs(make_rks(mol).kernel() - make_uks(mol).kernel()) < 1e-7


def test_gks_scalar_energy():
    mol = make_hf_mol()
    e_r = make_rks(mol).kernel()
    e_g = make_gks(mol).kernel()
    assert abs(e_r - e_g) < 1e-6


def test_gks_so_energy_differs():
    mol = make_hf_mol()
    e_sf = make_gks(mol).kernel()
    e_so = make_gks(mol, spin_orbit=True).kernel()
    assert abs(e_so - e_sf) > 1e-6


def test_rohf_closed_shell_energy():
    mol = make_hf_mol()
    e_r = make_rhf(mol).kernel()
    e_ro = make_rohf(mol).kernel()
    assert abs(e_r - e_ro) < 1e-8


def test_rohf_open_shell_runs():
    mol = pyscf.gto.M(atom='Li 0 0 0', basis='sto-3g', spin=1, verbose=0)
    mf = pyscf.scf.ROHF(mol).zora(grid_level=GRAD_GRID_LEVEL)
    mf.verbose = 0
    e = mf.kernel()
    assert mf.converged
    assert e < 0
