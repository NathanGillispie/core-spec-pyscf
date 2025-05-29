import pyscf.cvs
import pyscf
from pyscf.tdscf import TDHF, TDDFT

import numpy as np

def test_ghf_tdhf():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.GHF(mol)
    mf.kernel()

    tdobj = TDHF(mf)
    tdobj.kernel(nstates=80)
    e1 = tdobj.e[-16:]

    tdobj.core_idx=[0, 1]
    tdobj.kernel(nstates=16)
    e2 = tdobj.e[-16:]

    assert np.allclose(e1,e2)

def test_gks_tddft():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.GKS(mol)
    mf.xc = 'PBE0'
    mf.kernel()

    tdobj = TDDFT(mf)
    tdobj.kernel(nstates=80)
    e1 = tdobj.e[-16:]

    tdobj.core_idx=[0,1]
    tdobj.kernel(nstates=16)
    e2 = tdobj.e

    assert np.allclose(e1, e2, rtol=1e-3)

