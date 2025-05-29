#!/usr/bin/env python
import pyscf.cvs
import pyscf.zora
import pyscf
from pyscf.tdscf import TDHF

import numpy as np

def test_rhf_tdhf():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    tdobj = TDHF(mf)
    tdobj.kernel(nstates=22)
    e1 = tdobj.e[-4:]

    tdobj.core_idx=[0]
    tdobj.kernel(nstates=4)
    e2 = tdobj.e

    assert np.allclose(e1,e2,atol=1e-4)

