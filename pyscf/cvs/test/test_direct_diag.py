import pyscf
import pyscf.cvs
from pyscf.tdscf import TDA, TDDFT

import numpy as np
import pytest

@pytest.mark.parametrize("ref", [ "RKS", "UKS", "GKS", "RHF", "UHF", "GHF" ])
def test_direct_diag_tda(ref):
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()

    tdobj = TDA(mf)
    tdobj.kernel(nstates=100)
    e1 = tdobj.e

    tdobj.direct_diag = True
    tdobj.kernel()

    e2 = tdobj.e
    assert np.allclose(e1, e2)

@pytest.mark.parametrize("ref", [ "RKS", "UKS", "GKS", "RHF", "UHF", "GHF" ])
def test_direct_diag_rpa(ref):
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=2)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()

    tdobj = TDDFT(mf)
    tdobj.kernel(nstates=1000)
    e1 = tdobj.e

    tdobj.direct_diag = True
    tdobj.kernel()
    e2 = tdobj.e

    assert np.allclose(e1, e2)

test_direct_diag_rpa('RKS')
