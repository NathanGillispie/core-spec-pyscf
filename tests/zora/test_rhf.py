import numpy as np
import pyscf
import pyscf.zora
from math import isclose


def test_energy():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True)
    mf = pyscf.scf.RHF(mol)
    e1 = mf.kernel()
    mf = mf.zora()
    e2 = mf.kernel()
    assert isclose(e1, -128.47387687066833, abs_tol=1e-7)
    assert isclose(e2, -128.68095017915087, abs_tol=1e-7)


def test_energy_slow():
    mol = pyscf.M(atom='Zn 0 0 0', basis='6-31g', cart=True)
    mf = pyscf.scf.RHF(mol)
    e1 = mf.kernel()
    mf = mf.zora()
    e2 = mf.kernel()
    assert isclose(e1, -1777.4827533499622, abs_tol=1e-7)
    assert isclose(e2, -1801.2886699499340, abs_tol=1e-7)


def test_energy2():
    mol = pyscf.M(atom='H 0 0 0; H 1 0 0; H 2 0 0; H 3 0 0',
                  basis='6-31g',
                  cart=True)
    mf = pyscf.scf.RHF(mol)
    e1 = mf.kernel()
    mf = mf.zora()
    e2 = mf.kernel()
    assert isclose(e1, -2.1602439129951145, abs_tol=1e-7)
    assert isclose(e2, -2.1602716417644530, abs_tol=1e-7)


def test_zora_returns_mf():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = pyscf.scf.RHF(mol).zora()
    assert mf is not None
    assert hasattr(mf, 'with_zora')
    mf2 = mf.zora()
    assert mf2 is mf


def test_undo_zora_and_reset():
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1.1', basis='sto-3g', verbose=0)
    mf = pyscf.scf.RHF(mol).zora()
    h_zora = mf.get_hcore()
    mf_nr = mf.undo_zora()
    assert not hasattr(mf_nr, 'with_zora')
    h_nr = mf_nr.get_hcore()
    assert not np.allclose(h_zora, h_nr)

    mf = pyscf.scf.RHF(mol).zora()
    h0 = mf.get_hcore().copy()
    mol2 = mol.copy()
    coords = mol.atom_coords()
    coords[1, 2] += 0.2
    mol2.set_geom_(coords, unit='Bohr')
    mf.reset(mol2)
    h1 = mf.get_hcore()
    assert not np.allclose(h0, h1)
