import pyscf
import pyscf.cvs
from pyscf.tdscf import TDA, RPA

import numpy as np

import pytest

@pytest.mark.parametrize("ref", [ "RKS", "UKS", "GKS", "RHF", "UHF", "GHF" ])
def test_no_fxc_tda(ref):
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()

    tdobj = TDA(mf)
    tdobj.kernel(nstates=22)
    e1 = tdobj.e

    tdobj.no_fxc = True
    tdobj.kernel()
    e2 = tdobj.e
    assert np.allclose(e1, e2, atol=2e-6)

@pytest.mark.parametrize("ref", [ "RKS", "UKS", "GKS", "RHF", "UHF", "GHF" ])
def test_no_fxc_rpa(ref):
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()

    tdobj = RPA(mf)
    tdobj.kernel(nstates=22)
    e1 = tdobj.e

    tdobj.no_fxc = True
    tdobj.kernel()
    e2 = tdobj.e
    assert np.allclose(e1, e2, atol=2e-6)

