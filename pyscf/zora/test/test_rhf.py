import numpy as np
import pytest
import pyscf
import pyscf.zora

def test_energy():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True)
    mf = pyscf.scf.RHF(mol)
    e1 = mf.kernel()
    mf.zora()
    e2 = mf.kernel()
    assert e2 < e1

def test_energy_slow():
    mol = pyscf.M(atom='Zn 0 0 0', basis='6-31g', cart=True)
    mf = pyscf.scf.RHF(mol)
    e1 = mf.kernel()
    mf.zora()
    e2 = mf.kernel()
    assert e2 < e1

def test_energy2():
    mol = pyscf.M(atom='H 0 0 0; H 1 0 0; H 2 0 0; H 3 0 0', basis='6-31g', cart=True)
    mf = pyscf.scf.RHF(mol)
    e1 = mf.kernel()
    mf.zora()
    e2 = mf.kernel()
    assert e2 < e1
