from pathlib import Path

import pyscf
import pyscf.cvs
from pyscf import dft, scf
from pyscf.tdscf import TDA, RPA
from pyscf.cvs.td import _get_ab

import numpy as np

import pytest


_REFERENCE = Path(__file__).with_name("no_fxc_reference.npz")
_REFS = ("RHF", "UHF", "GHF", "RKS", "UKS", "GKS")


def _build_mf(ref, reference=None):
    mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74',
                   basis='3-21g', verbose=0)
    if ref in ("RKS", "UKS", "GKS"):
        mf = getattr(dft, ref)(mol, xc='LDA').run()
    else:
        mf = getattr(scf, ref)(mol).run()
    if reference is not None:
        mf.mo_coeff = reference[f"{ref}_mo_coeff"]
        mf.mo_energy = reference[f"{ref}_mo_energy"]
        mf.mo_occ = reference[f"{ref}_mo_occ"]
    return mf


@pytest.fixture(scope="module")
def no_fxc_reference():
    with np.load(_REFERENCE) as data:
        yield {key: data[key] for key in data.files}


def _assert_archived_ab(no_fxc_reference, ref, A, B):
    if isinstance(A, tuple):
        for spin, matrix in enumerate(A):
            np.testing.assert_allclose(
                matrix, no_fxc_reference[f"{ref}_A_{spin}"], atol=1e-12)
        for spin, matrix in enumerate(B):
            np.testing.assert_allclose(
                matrix, no_fxc_reference[f"{ref}_B_{spin}"], atol=1e-12)
    else:
        np.testing.assert_allclose(
            A, no_fxc_reference[f"{ref}_A"], atol=1e-12)
        np.testing.assert_allclose(
            B, no_fxc_reference[f"{ref}_B"], atol=1e-12)


@pytest.mark.parametrize("ref", _REFS)
def test_no_fxc_ab_matches_archived_output(no_fxc_reference, ref):
    mf = _build_mf(ref, no_fxc_reference)
    tdobj = TDA(mf).cvs(no_fxc=True)
    A, B = _get_ab(tdobj, no_fxc=True)
    _assert_archived_ab(no_fxc_reference, ref, A, B)


@pytest.mark.parametrize("ref", _REFS)
@pytest.mark.parametrize("td_cls", [TDA, RPA])
def test_no_fxc_kernel_matches_archived_output(
        no_fxc_reference, ref, td_cls):
    mf = _build_mf(ref, no_fxc_reference)
    tdobj = td_cls(mf).cvs(no_fxc=True, direct_diag=True)
    tdobj.kernel(nstates=3)

    assert tdobj.direct_diag is True
    np.testing.assert_allclose(
        tdobj.e, no_fxc_reference[f"{ref}_{td_cls.__name__.lower()}_e"],
        atol=1e-9)


@pytest.mark.parametrize("ref", ["RKS", "UKS"])
@pytest.mark.parametrize("td_cls", [TDA, RPA])
def test_no_fxc_iterative_kernel_matches_archived_output(
        no_fxc_reference, ref, td_cls):
    mf = _build_mf(ref, no_fxc_reference)
    tdobj = td_cls(mf).cvs(no_fxc=True, direct_diag=False)
    tdobj.kernel(nstates=3)

    assert tdobj.direct_diag is False
    np.testing.assert_allclose(
        tdobj.e, no_fxc_reference[f"{ref}_{td_cls.__name__.lower()}_e"],
        atol=1e-9)


@pytest.mark.parametrize("td_cls", [TDA, RPA])
def test_no_fxc_lih_direct_matches_davidson(td_cls):
    mol = pyscf.M(atom='Li 0 0 0; H 0 0 1.6',
                   basis='def2-SVP', verbose=0)
    mf = dft.RKS(mol, xc='PBE').run()

    direct = td_cls(mf).cvs(
        core_idx=[0], no_fxc=True, direct_diag=True)
    iterative = td_cls(mf).cvs(
        core_idx=[0], no_fxc=True, direct_diag=False)
    direct.kernel(nstates=4)
    iterative.kernel(nstates=4)

    assert direct.e.shape == iterative.e.shape == (4,)
    np.testing.assert_allclose(direct.e, iterative.e, atol=1e-8)


def test_no_fxc_keeps_mean_field_unchanged(monkeypatch):
    mf = _build_mf("RKS")
    original_xc = mf.xc
    to_hf_calls = []
    original_to_hf = mf.to_hf

    def to_hf():
        to_hf_calls.append(True)
        return original_to_hf()

    monkeypatch.setattr(mf, "to_hf", to_hf)
    tdobj = TDA(mf).cvs(no_fxc=True, direct_diag=False)
    tdobj.kernel(nstates=1)

    assert len(to_hf_calls) == 1
    assert tdobj._scf is mf
    assert mf.xc == original_xc


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
    mol = pyscf.M(atom='H 0 0 0; Li 0 0 1.1', basis='def2-SVP', verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.xc = 'LDA'
    mf.kernel()

    td_dft = TDA(mf)
    td_dft.kernel(nstates=2)
    td_hf = TDA(mf).cvs(no_fxc=True, direct_diag=True)
    td_hf.kernel(nstates=2)
    assert td_hf.e.shape == td_dft.e.shape
    assert not np.allclose(td_dft.e, td_hf.e)


@pytest.mark.parametrize("ref", ["RKS", "UKS", "GKS"])
def test_no_fxc_ks_rpa_runs(ref):
    mol = pyscf.M(atom='H 0 0 0; Li 0 0 1.1', basis='def2-SVP', verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.xc = 'LDA'
    mf.kernel()

    tdobj = RPA(mf).cvs(no_fxc=True, direct_diag=True)
    tdobj.kernel(nstates=1)
    assert tdobj.e is not None
    assert np.all(np.isfinite(tdobj.e))
