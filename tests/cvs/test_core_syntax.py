'''
The purpose of this test is to test the syntax of core_idx / .cvs().
'''
import numpy
import pyscf
import pyscf.cvs
from pyscf.tdscf import RPA, TDA
from pyscf.cvs.td import CVS

import pytest

testdata = [("RKS", [0]), ("UKS", ([0], [0])), ("GKS", [0]), ("RHF", [0]),
            ("UHF", ([0], [0])), ("GHF", [0])]


def _copy_occ(mf):
    occ = mf.mo_occ
    if isinstance(occ, numpy.ndarray) and occ.ndim == 1:
        return occ.copy(), numpy.asarray(mf.mo_coeff).copy(), numpy.asarray(
            mf.mo_energy).copy()
    return (occ[0].copy(), occ[1].copy()), (mf.mo_coeff[0].copy(),
                                            mf.mo_coeff[1].copy()), (
                                                mf.mo_energy[0].copy(),
                                                mf.mo_energy[1].copy())


def _assert_scf_unchanged(mf, occ0, coeff0, energy0, nelec0):
    occ, coeff, energy = _copy_occ(mf)
    if isinstance(occ0[0], numpy.ndarray):
        assert numpy.allclose(occ[0], occ0[0])
        assert numpy.allclose(occ[1], occ0[1])
        assert numpy.allclose(coeff[0], coeff0[0])
        assert numpy.allclose(coeff[1], coeff0[1])
        assert numpy.allclose(energy[0], energy0[0])
        assert numpy.allclose(energy[1], energy0[1])
    else:
        assert numpy.allclose(occ, occ0)
        assert numpy.allclose(coeff, coeff0)
        assert numpy.allclose(energy, energy0)
    assert mf.mol.nelec == nelec0


@pytest.mark.parametrize("ref,core_idx", testdata)
def test_tda(ref, core_idx):
    mol = pyscf.M(atom='H 0 0 0; H .5 0 0', basis='sto-3g', verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()
    occ0, coeff0, energy0 = _copy_occ(mf)
    nelec0 = mol.nelec
    tdobj = TDA(mf).cvs(core_idx=core_idx)
    assert isinstance(tdobj, CVS)
    tdobj.kernel(nstates=1)
    assert tdobj.frozen is not None
    _assert_scf_unchanged(mf, occ0, coeff0, energy0, nelec0)

    tdobj = tdobj.cvs()
    tdobj.core_valence(core_idx=core_idx)
    tdobj.kernel(nstates=1)
    _assert_scf_unchanged(mf, occ0, coeff0, energy0, nelec0)

    tdobj2 = TDA(mf).cvs()
    tdobj2.kernel(core_idx=core_idx, nstates=1)
    _assert_scf_unchanged(mf, occ0, coeff0, energy0, nelec0)


@pytest.mark.parametrize("ref,core_idx", testdata)
def test_rpa(ref, core_idx):
    mol = pyscf.M(atom='H 0 0 0; H .5 0 0', basis='sto-3g', verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()
    occ0, coeff0, energy0 = _copy_occ(mf)
    nelec0 = mol.nelec
    tdobj = RPA(mf).cvs(core_idx=core_idx)
    tdobj.kernel(nstates=1)
    assert tdobj.frozen is not None
    _assert_scf_unchanged(mf, occ0, coeff0, energy0, nelec0)

    tdobj.core_valence(core_idx=core_idx)
    tdobj.kernel(nstates=1)
    _assert_scf_unchanged(mf, occ0, coeff0, energy0, nelec0)
