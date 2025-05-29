import pyscf
import pyscf.cvs
from pyscf.tdscf import TDA

import numpy as np

def test_rhf_tda():
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

def test_uks_tda():
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

def test_ghf_tda():
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

