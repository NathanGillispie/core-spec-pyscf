import pyscf.cvs
import pyscf
from pyscf.tdscf import TDHF, TDDFT

import numpy as np

def test_uhf_tdhf():
    mol = pyscf.M(atom='Cl 0 0 0', basis='6-31g', cart=True, spin=1, verbose=0)
    mf = pyscf.scf.UHF(mol)
    mf.kernel()

    tdobj = TDHF(mf)
    tdobj.kernel(nstates=74)
    e1 = tdobj.e[-6:]

    tdobj.core_idx=([0],[0])
    tdobj.kernel(nstates=13)
    e2 = tdobj.e[-6:]

    assert np.allclose(e1,e2,atol=1e-4)

def test_uks_tddft():
    mol = pyscf.M(atom='Cl 0 0 0', basis='6-31g', cart=True, spin=1, verbose=0)
    mf = pyscf.scf.UKS(mol)
    mf.xc = 'PBE0'
    mf.kernel()

    tdobj = TDDFT(mf)
    tdobj.kernel(nstates=80)
    e1 = tdobj.e[-9:]

    tdobj.core_idx=([1,2],[0,1])
    tdobj.kernel(nstates=24)
    e2 = tdobj.e[-9:]

    assert np.allclose(e1, e2, rtol=4e-5)

