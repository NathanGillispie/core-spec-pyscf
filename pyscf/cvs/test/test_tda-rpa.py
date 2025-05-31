import pyscf
import pyscf.cvs
from pyscf.tdscf import RPA, TDA

import numpy as np

def test_rhf_RPA():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    tdobj = RPA(mf)
    tdobj.kernel(nstates=22)
    e1 = tdobj.e[-4:]

    tdobj.core_idx=[0]
    tdobj.kernel(nstates=4)
    e2 = tdobj.e

    assert np.allclose(e1,e2,atol=1e-4)

def test_rks_RPA():
    mol = pyscf.M(atom='Ar 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.RKS(mol)
    mf.xc = 'PBE0'
    mf.kernel()

    tdobj = TDA(mf)
    tdobj.kernel(nstates=36)
    e1 = tdobj.e[-4:]

    tdobj.core_idx=[0]
    tdobj.kernel(nstates=4)
    e2 = tdobj.e

    assert np.allclose(e1, e2, rtol=5e-5)

def test_uhf_RPA():
    mol = pyscf.M(atom='Cl 0 0 0', basis='6-31g', cart=True, spin=1, verbose=0)
    mf = pyscf.scf.UHF(mol)
    mf.kernel()

    tdobj = RPA(mf)
    tdobj.kernel(nstates=74)
    e1 = tdobj.e[-6:]

    tdobj.core_idx=([0],[0])
    tdobj.kernel(nstates=13)
    e2 = tdobj.e[-6:]

    assert np.allclose(e1,e2,atol=1e-4)

def test_uks_RPA():
    mol = pyscf.M(atom='Cl 0 0 0', basis='6-31g', cart=True, spin=1, verbose=0)
    mf = pyscf.scf.UKS(mol)
    mf.xc = 'PBE0'
    mf.kernel()

    tdobj = RPA(mf)
    tdobj.kernel(nstates=80)
    e1 = tdobj.e[-9:]

    tdobj.core_idx=([1,2],[0,1])
    tdobj.kernel(nstates=24)
    e2 = tdobj.e[-9:]

    assert np.allclose(e1, e2, rtol=4e-5)

def test_ghf_RPA():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.GHF(mol)
    mf.kernel()

    tdobj = RPA(mf)
    tdobj.kernel(nstates=80)
    e1 = tdobj.e[-16:]

    tdobj.core_idx=[0, 1]
    tdobj.kernel(nstates=16)
    e2 = tdobj.e[-16:]

    assert np.allclose(e1,e2)

def test_gks_RPA():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.GKS(mol)
    mf.xc = 'PBE0'
    mf.kernel()

    tdobj = RPA(mf)
    tdobj.kernel(nstates=80)
    e1 = tdobj.e[-16:]

    tdobj.core_idx=[0,1]
    tdobj.kernel(nstates=16)
    e2 = tdobj.e

    assert np.allclose(e1, e2, rtol=1e-3)


def test_rhf_TDA():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    tdobj = TDA(mf)
    tdobj.kernel(nstates=20)
    e1 = tdobj.e[-4:]

    tdobj.core_idx=[0]
    tdobj.kernel(nstates=4)
    e2 = tdobj.e

    assert np.allclose(e1, e2, rtol=4e-5)

def test_uks_TDA():
    mol = pyscf.M(atom='Cl 0 0 0', basis='6-31g', cart=True, spin=1, verbose=0)
    mf = pyscf.scf.UKS(mol)
    mf.xc = 'PBE0'
    mf.kernel()

    tdobj = TDA(mf)
    tdobj.kernel(nstates=80)
    e1 = tdobj.e[-9:]

    tdobj.core_idx=([1,2],[0,1])
    tdobj.kernel(nstates=24)
    e2 = tdobj.e[-9:]

    assert np.allclose(e1, e2, rtol=4e-5)

def test_ghf_TDA():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.GHF(mol)
    mf.kernel()

    tdobj = TDA(mf)
    tdobj.kernel(nstates=80)
    e1 = tdobj.e[-16:]

    tdobj.core_idx=[0, 1]
    tdobj.kernel(nstates=16)
    e2 = tdobj.e

    assert np.allclose(e1,e2,rtol=2e-5)

