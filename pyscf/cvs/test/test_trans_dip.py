import pyscf
from pyscf.tdscf import TDA, RPA
import pyscf.cvs

import numpy as np
np.set_printoptions(precision=5, suppress=True, linewidth=200)
import pytest

from pyscf.lib import StreamObject
StreamObject.check_sanity = lambda self, *args, **kwargs: None

@pytest.mark.parametrize("ref", [ "RHF", "RKS"])#, "UKS", "UHF"])
def test_trans_dip_tda(ref):
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()

    tdobj = TDA(mf)
    tdobj.nstates=30
    tdobj.kernel()
    e1 = tdobj.e
    t1 = np.sum(tdobj.transition_dipole()**2, axis=1)

    tdobj.direct_diag = True
    tdobj.kernel()
    e2 = tdobj.e
    t2 = np.sum(tdobj.transition_dipole()**2, axis=1)

    assert np.allclose(e1, e2)
    assert np.allclose(t1, t2)

@pytest.mark.parametrize("ref", [ "RHF", "RKS"])#, "UKS", "UHF"])
def test_trans_dip_rpa(ref):
    mol = pyscf.M(atom='Ne 0 0 0', basis='3-21g', cart=True, verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()

    tdobj = RPA(mf)
    tdobj.nstates=1000
    tdobj.kernel()
    e1 = tdobj.e
    t1 = np.sum(tdobj.transition_dipole()**2, axis=1)

    tdobj.direct_diag = True
    tdobj.kernel()
    e2 = tdobj.e
    t2 = np.sum(tdobj.transition_dipole()**2, axis=1)

    assert np.allclose(e1, e2)
    assert np.allclose(t1, t2)


# FAILING WHILE WORKING

@pytest.mark.xfail
@pytest.mark.parametrize("ref", [ "UKS", "UHF"])
def test_trans_dip_rpa(ref):
    mol = pyscf.M(atom='Ne 0 0 0', basis='3-21g', cart=True, verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()

    tdobj = RPA(mf)
    tdobj.nstates=1000
    tdobj.kernel()
    e1 = tdobj.e
    t1 = np.sum(tdobj.transition_dipole()**2, axis=1)

    tdobj.direct_diag = True
    tdobj.kernel()
    e2 = tdobj.e
    t2 = np.sum(tdobj.transition_dipole()**2, axis=1)

    assert np.allclose(e1, e2)
    assert np.allclose(t1, t2)

@pytest.mark.xfail
@pytest.mark.parametrize("ref", [ "UKS", "UHF"])
def test_trans_dip_tda_xfail(ref):
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()

    tdobj = TDA(mf)
    tdobj.nstates=30
    tdobj.kernel()
    e1 = tdobj.e
    t1 = np.sum(tdobj.transition_dipole()**2, axis=1)

    tdobj.direct_diag = True
    tdobj.kernel()
    e2 = tdobj.e
    t2 = np.sum(tdobj.transition_dipole()**2, axis=1)

    assert np.allclose(e1, e2)
    assert np.allclose(t1, t2)
