import numpy as np
import pytest
import pyscf
import pyscf.zora
from math import isclose

def test_energy():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True)
    mf = pyscf.scf.RHF(mol)
    e1 = mf.kernel()
    mf.zora()
    e2 = mf.kernel()
    assert isclose(e1, -128.47387687066833, abs_tol=1e-7)
    assert isclose(e2, -128.68095017915087, abs_tol=1e-7)


def test_energy_slow():
    mol = pyscf.M(atom='Zn 0 0 0', basis='6-31g', cart=True)
    mf = pyscf.scf.RHF(mol)
    e1 = mf.kernel()
    mf.zora()
    e2 = mf.kernel()
    assert isclose(e1, -1777.4827533499622, abs_tol=1e-7)
    assert isclose(e2, -1801.2886699499340, abs_tol=1e-7)

def test_energy2():
    mol = pyscf.M(atom='H 0 0 0; H 1 0 0; H 2 0 0; H 3 0 0', basis='6-31g', cart=True)
    mf = pyscf.scf.RHF(mol)
    e1 = mf.kernel()
    mf.zora()
    e2 = mf.kernel()
    assert isclose(e1, -2.1602439129951145, abs_tol=1e-7)
    assert isclose(e2, -2.1602716417644530, abs_tol=1e-7)

