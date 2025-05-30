import pyscf
import pyscf.cvs
from pyscf.tdscf import TDA

import numpy as np

import pytest

@pytest.mark.parametrize("ref", [ "RKS", "UKS", "GKS", "RHF", "UHF", "GHF" ])
def test_direct_diag_tda(ref):
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()

    tdobj = TDA(mf)
    tdobj.kernel(nstates=22)
    e1 = tdobj.e

    tdobj.direct_diag = True
    tdobj.kernel()
    e2 = tdobj.e
    assert np.allclose(e1, e2)

def test_rhf_direct_diag():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    tdobj = TDA(mf)
    tdobj.kernel(nstates=1200)
    e1 = tdobj.e

    tdobj.direct_diag = True
    tdobj.kernel()
    e2 = tdobj.e
    assert np.allclose(e1, e2)

test_rhf_direct_diag()
