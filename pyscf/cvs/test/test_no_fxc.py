import pyscf
import pyscf.cvs
from pyscf.tdscf import TDA, RPA

import numpy as np

import pytest


@pytest.mark.parametrize("ref", ["RHF", "UHF", "GHF"])
def test_no_fxc_hf_matches_tda(ref):
    mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='3-21g', verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()

    tdobj = TDA(mf)
    tdobj.kernel(nstates=2)
    e1 = tdobj.e.copy()

    tdobj = tdobj.cvs(no_fxc=True)
    tdobj.kernel(nstates=2)
    assert np.allclose(e1, tdobj.e)


@pytest.mark.parametrize("ref", ["RHF", "UHF", "GHF"])
def test_no_fxc_hf_matches_rpa(ref):
    mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='3-21g', verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()

    tdobj = RPA(mf)
    tdobj.kernel(nstates=2)
    e1 = tdobj.e.copy()

    tdobj = tdobj.cvs(no_fxc=True)
    tdobj.kernel(nstates=2)
    assert np.allclose(e1, tdobj.e)


@pytest.mark.parametrize("ref", ["RKS", "UKS", "GKS"])
def test_no_fxc_ks_tda_runs(ref):
    mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='3-21g', verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.xc = 'PBE'
    mf.kernel()

    td_dft = TDA(mf)
    td_dft.kernel(nstates=2)
    td_hf = TDA(mf).cvs(no_fxc=True)
    td_hf.kernel(nstates=2)
    assert td_hf.e.shape == td_dft.e.shape
    assert not np.allclose(td_dft.e, td_hf.e)


@pytest.mark.parametrize("ref", ["RKS", "UKS", "GKS"])
def test_no_fxc_ks_rpa_runs(ref):
    mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='3-21g', verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.xc = 'PBE'
    mf.kernel()

    tdobj = RPA(mf).cvs(no_fxc=True)
    tdobj.kernel(nstates=1)
    assert tdobj.e is not None
    assert np.all(np.isfinite(tdobj.e))
