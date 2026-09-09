import pyscf
import pyscf.cvs
from pyscf.tdscf import RPA, TDA

import numpy as np


def _frozen_rhf(mf, core_idx):
    occ = np.where(mf.mo_occ != 0)[0]
    return np.setdiff1d(occ, np.atleast_1d(core_idx)).tolist()


def test_rhf_RPA():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    tdobj = RPA(mf)
    tdobj.kernel(nstates=22)
    e1 = tdobj.e[-4:]

    tdobj = tdobj.cvs(core_idx=[0])
    tdobj.kernel(nstates=4)
    e2 = tdobj.e

    assert np.allclose(e1, e2, atol=1e-4)

    td_frz = RPA(mf, frozen=_frozen_rhf(mf, [0]))
    td_frz.kernel(nstates=4)
    assert np.allclose(e2, td_frz.e)


def test_rks_RPA():
    mol = pyscf.M(atom='Ar 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.RKS(mol)
    mf.xc = 'PBE0'
    mf.kernel()

    tdobj = TDA(mf)
    tdobj.kernel(nstates=36)
    e1 = tdobj.e[-4:]

    tdobj = tdobj.cvs(core_idx=[0])
    tdobj.kernel(nstates=4)
    e2 = tdobj.e

    assert np.allclose(e1, e2, rtol=5e-5)


def test_uhf_RPA():
    mol = pyscf.M(atom='Cl 0 0 0', basis='6-31g', cart=True, spin=1, verbose=0)
    mf = pyscf.scf.UHF(mol)
    mf.kernel()

    core_idx = ([0], [0])
    tdobj = RPA(mf).cvs(core_idx=core_idx)
    tdobj.kernel(nstates=6)
    e_cvs = tdobj.e

    td_frz = RPA(mf, frozen=tdobj.frozen)
    td_frz.kernel(nstates=6)
    assert np.allclose(e_cvs, td_frz.e, atol=1e-6)


def test_uks_RPA():
    mol = pyscf.M(atom='Cl 0 0 0', basis='6-31g', cart=True, spin=1, verbose=0)
    mf = pyscf.scf.UKS(mol)
    mf.xc = 'PBE0'
    mf.kernel()

    core_idx = ([1, 2], [0, 1])
    tdobj = RPA(mf).cvs(core_idx=core_idx)
    tdobj.kernel(nstates=8)
    e_cvs = tdobj.e

    td_frz = RPA(mf, frozen=tdobj.frozen)
    td_frz.kernel(nstates=8)
    assert np.allclose(e_cvs, td_frz.e, rtol=1e-6)


def test_ghf_RPA():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.GHF(mol)
    mf.kernel()

    tdobj = RPA(mf)
    tdobj.kernel(nstates=80)
    e1 = tdobj.e[-16:]

    tdobj = tdobj.cvs(core_idx=[0, 1])
    tdobj.kernel(nstates=16)
    e2 = tdobj.e[-16:]

    assert np.allclose(e1, e2)

    td_frz = RPA(mf, frozen=_frozen_rhf(mf, [0, 1]))
    td_frz.kernel(nstates=16)
    assert np.allclose(e2, td_frz.e)


def test_gks_RPA():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.GKS(mol)
    mf.xc = 'PBE0'
    mf.kernel()

    tdobj = RPA(mf)
    tdobj.kernel(nstates=80)
    e1 = tdobj.e[-16:]

    tdobj = tdobj.cvs(core_idx=[0, 1])
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

    tdobj = tdobj.cvs(core_idx=[0])
    tdobj.kernel(nstates=4)
    e2 = tdobj.e

    assert np.allclose(e1, e2, rtol=4e-5)


def test_uks_TDA():
    mol = pyscf.M(atom='Cl 0 0 0', basis='6-31g', cart=True, spin=1, verbose=0)
    mf = pyscf.scf.UKS(mol)
    mf.xc = 'PBE0'
    mf.kernel()

    core_idx = ([1, 2], [0, 1])
    tdobj = TDA(mf).cvs(core_idx=core_idx)
    tdobj.kernel(nstates=8)
    e_cvs = tdobj.e

    td_frz = TDA(mf, frozen=tdobj.frozen)
    td_frz.kernel(nstates=8)
    assert np.allclose(e_cvs, td_frz.e, rtol=1e-6)


def test_ghf_TDA():
    mol = pyscf.M(atom='Ne 0 0 0', basis='6-31g', cart=True, verbose=0)
    mf = pyscf.scf.GHF(mol)
    mf.kernel()

    tdobj = TDA(mf)
    tdobj.kernel(nstates=80)
    e1 = tdobj.e[-16:]

    tdobj = tdobj.cvs(core_idx=[0, 1])
    tdobj.kernel(nstates=16)
    e2 = tdobj.e

    assert np.allclose(e1, e2, rtol=2e-5)
