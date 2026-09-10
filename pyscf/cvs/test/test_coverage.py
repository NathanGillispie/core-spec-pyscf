'''Edge-case and coverage tests for pyscf.cvs.'''
import sys

import numpy
import pytest
from pyscf.tdscf import TDA, RPA

import pyscf
import pyscf.cvs
from pyscf.cvs import no_fxc
from pyscf.cvs import rhf as cvs_rhf
from pyscf.cvs import uhf as cvs_uhf
from pyscf.cvs import ghf as cvs_ghf
from pyscf.cvs.td import CVS, _get_ab
from pyscf.cvs.no_fxc import (
    get_ab_no_fxc_ghf,
    get_ab_no_fxc_rhf,
    get_ab_no_fxc_uhf,
)


def _he(ref, xc=None):
    mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='3-21g', verbose=0)
    mf = eval(f'pyscf.scf.{ref}(mol)')
    if xc is not None:
        mf.xc = xc
    mf.kernel()
    return mf


class _FakeTD:
    verbose = 0
    stdout = sys.stdout
    nstates = 1
    positive_eig_threshold = 1e-5
    chkfile = None
    frozen = None
    _scf = None

    def check_sanity(self):
        return self

    def dump_flags(self):
        return self

    def _finalize(self):
        return self


def test_core_valence_int_and_missing():
    mf = _he('RHF')
    td = TDA(mf).cvs()
    with pytest.raises(RuntimeError, match='Core orbitals not specified'):
        td.core_valence()
    td.core_valence(core_idx=0)
    occ = numpy.where(mf.mo_occ != 0)[0]
    assert td.frozen == [int(i) for i in occ if i != 0]

    td.core_valence(core_idx=numpy.int64(0))
    assert td.core_idx == [0]


def test_core_window_restricted_is_lazy_and_matches_indices():
    mol = pyscf.M(atom='Be 0 0 0', basis='sto-3g', verbose=0)
    mf = pyscf.scf.RHF(mol).run()
    occ = numpy.where(mf.mo_occ != 0)[0]
    energy = mf.mo_energy[occ[0]]
    window = (energy - 1e-10, energy + 1e-10)

    td = TDA(mf).cvs(core_window=window)
    assert td.core_idx is None
    td.core_valence()
    assert td.core_idx == [int(occ[0])]
    assert td.frozen == [int(i) for i in occ[1:]]

    td.kernel(nstates=1)
    td_idx = TDA(mf).cvs(core_idx=[int(occ[0])])
    td_idx.kernel(nstates=1)
    assert numpy.allclose(td.e, td_idx.e)


def test_core_window_unrestricted_shared_and_spin_specific():
    mf = _he('UHF')
    occ_a = numpy.where(mf.mo_occ[0] != 0)[0]
    occ_b = numpy.where(mf.mo_occ[1] != 0)[0]
    window = (
        min(mf.mo_energy[0][occ_a[0]], mf.mo_energy[1][occ_b[0]]) - 1e-10,
        max(mf.mo_energy[0][occ_a[0]], mf.mo_energy[1][occ_b[0]]) + 1e-10,
    )
    td = TDA(mf).cvs(core_window=window)
    td.core_valence()
    assert td.core_idx == ([int(occ_a[0])], [int(occ_b[0])])

    spin_window = (
        (mf.mo_energy[0][occ_a[0]] - 1e-10,
         mf.mo_energy[0][occ_a[0]] + 1e-10),
        (mf.mo_energy[1][occ_b[0]] - 1e-10,
         mf.mo_energy[1][occ_b[0]] + 1e-10),
    )
    td = TDA(mf).cvs(core_window=spin_window)
    td.core_valence()
    assert td.core_idx == ([int(occ_a[0])], [int(occ_b[0])])


def test_core_window_errors():
    mf = _he('RHF')
    td = TDA(mf).cvs()
    with pytest.raises(ValueError, match='either core_idx or core_window'):
        td.core_valence(core_idx=[0], core_window=(-1, 1))
    with pytest.raises(ValueError, match='emin <= emax'):
        td.core_valence(core_window=(1, -1))
    with pytest.raises(ValueError, match='does not contain an occupied'):
        td.core_valence(core_window=(100, 101))

    mf_u = _he('UHF')
    td_u = TDA(mf_u).cvs()
    with pytest.raises(ValueError, match='UHF core_window'):
        td_u.core_valence(core_window=(1, 2, 3))


def test_core_valence_unoccupied_warns():
    mf = _he('RHF')
    td = TDA(mf).cvs()
    td.verbose = 4
    td.core_valence(core_idx=[0, 99])
    occ = numpy.where(mf.mo_occ != 0)[0]
    assert td.frozen == [int(i) for i in occ if i != 0]


def test_uhf_core_valence_errors():
    mf = _he('UHF')
    td = TDA(mf).cvs()
    with pytest.raises(RuntimeError, match='Core orbitals not specified'):
        td.core_valence()
    with pytest.raises(ValueError, match='idx_alpha'):
        td.core_valence(core_idx=[0])
    with pytest.raises(ValueError, match='idx_alpha'):
        td.core_valence(core_idx=0)
    td.core_valence(core_idx=(0, 0))
    assert td.core_idx == ([0], [0])
    td.verbose = 4
    td.core_valence(core_idx=([0], [99]))


def test_pad_frozen_uhf_both_directions():
    from pyscf.cvs._utils import _pad_frozen_uhf
    mo_occ = (numpy.array([1., 1., 0., 0.]), numpy.array([1., 0., 0., 0.]))
    fa, fb = _pad_frozen_uhf(mo_occ, [1], [])
    assert len(fa) == len(fb)
    fa, fb = _pad_frozen_uhf(mo_occ, [], [1])
    assert len(fa) == len(fb)
    fa, fb = _pad_frozen_uhf(mo_occ, [1], [0])
    assert len(fa) == len(fb)


def test_ghf_core_valence_int():
    mf = _he('GHF')
    td = TDA(mf).cvs()
    td.core_valence(core_idx=0)
    assert td.core_idx == [0]


def test_cvs_wrap_rewrapped_and_undo():
    mf = _he('RHF')
    td = TDA(mf)
    wrapped = td.cvs(core_idx=[0], direct_diag=True)
    assert wrapped is not td
    assert not isinstance(td, CVS)
    assert isinstance(wrapped, CVS)
    again = wrapped.cvs(core_idx=[0], no_fxc=True)
    assert again is wrapped
    assert again.no_fxc is True
    again.cvs(direct_diag=False)
    assert again.direct_diag is False
    wrapped.dump_flags()
    plain = wrapped.undo_cvs()
    assert not isinstance(plain, CVS)


def test_kernel_kwargs_direct_diag_nstates_chkfile(tmp_path):
    mf = _he('RHF')
    td = TDA(mf).cvs(direct_diag=True)
    td.chkfile = str(tmp_path / 'tda.chk')
    e, xy = td.kernel(nstates=1)
    assert len(e) == 1
    assert (tmp_path / 'tda.chk').exists()

    td = RPA(mf).cvs(direct_diag=True)
    td.chkfile = str(tmp_path / 'rpa.chk')
    e, xy = td.kernel(nstates=1)
    assert len(e) == 1


@pytest.mark.parametrize('ref', ['UHF', 'GHF'])
def test_chkfile_uhf_ghf(tmp_path, ref):
    mf = _he(ref)
    td = TDA(mf).cvs(direct_diag=True)
    td.chkfile = str(tmp_path / f'{ref}_tda.chk')
    td.kernel(nstates=1)
    assert (tmp_path / f'{ref}_tda.chk').exists()

    td = RPA(mf).cvs(direct_diag=True)
    td.chkfile = str(tmp_path / f'{ref}_rpa.chk')
    td.kernel(nstates=1)
    assert (tmp_path / f'{ref}_rpa.chk').exists()


def test_no_fxc_get_ab_requires_mf():
    with pytest.raises(NotImplementedError):
        get_ab_no_fxc_rhf()
    with pytest.raises(NotImplementedError):
        get_ab_no_fxc_uhf()
    with pytest.raises(NotImplementedError):
        get_ab_no_fxc_ghf()


def test_no_fxc_frozen_shapes():
    mol = pyscf.M(atom='Be 0 0 0', basis='sto-3g', verbose=0)
    mf = pyscf.scf.RHF(mol).run()
    a, b = get_ab_no_fxc_rhf(mf)
    nocc = int((mf.mo_occ == 2).sum())
    nvir = int((mf.mo_occ == 0).sum())
    assert a.shape == (nocc, nvir, nocc, nvir)

    a1, b1 = get_ab_no_fxc_rhf(mf, frozen=0)
    assert a1.shape == a.shape
    occ = numpy.where(mf.mo_occ != 0)[0]
    a2, _ = get_ab_no_fxc_rhf(mf, frozen=occ[1:].tolist())
    assert a2.shape[0] == 1

    with pytest.raises(NotImplementedError):
        no_fxc._mask_restricted(1.5, 4)

    mf_u = pyscf.scf.UHF(mol).run()
    au, bu = get_ab_no_fxc_uhf(mf_u)
    au2, _ = get_ab_no_fxc_uhf(mf_u, frozen=1)
    au3, _ = get_ab_no_fxc_uhf(mf_u, frozen=[0])
    au4, _ = get_ab_no_fxc_uhf(mf_u, frozen=())
    assert au4[0].shape == au[0].shape
    au5, _ = get_ab_no_fxc_uhf(mf_u, frozen=([], []))
    with pytest.raises(NotImplementedError):
        no_fxc._mask_unrestricted(1.5, 4, 4)

    mf_g = _he('GHF')
    ag, bg = get_ab_no_fxc_ghf(mf_g, frozen=None)
    assert ag.shape[0] >= 1
    mf_g.mo_coeff = mf_g.mo_coeff.astype(complex)
    agc, bgc = get_ab_no_fxc_ghf(mf_g)
    assert agc.dtype == numpy.complex128


def test_ghf_get_ab_forwards_frozen():
    mf = _he('GHF')
    td = TDA(mf).cvs(core_idx=[0])
    a, b = td.get_ab()
    from pyscf.tdscf.ghf import get_ab as ghf_get_ab
    a2, b2 = ghf_get_ab(mf, frozen=td.frozen)
    assert numpy.allclose(a, a2)
    assert a.shape[0] == 1


def test_rhf_get_ab_super():
    mf = _he('RHF')
    td = TDA(mf).cvs()
    a, b = td.get_ab()
    a2, b2 = td.get_ab(mf=mf, frozen=None)
    assert a.shape == a2.shape


def _rhf_ab(shift=0.0):
    A = numpy.zeros((1, 1, 1, 1))
    B = numpy.zeros((1, 1, 1, 1))
    A[0, 0, 0, 0] = 2.0 + shift
    B[0, 0, 0, 0] = 0.4
    return A, B


def _uhf_ab():
    A = numpy.zeros((1, 1, 1, 1))
    B = numpy.zeros((1, 1, 1, 1))
    A[0, 0, 0, 0] = 2.0
    B[0, 0, 0, 0] = 0.4
    z = numpy.zeros((1, 1, 1, 1))
    return (A, z, A.copy()), (B, z.copy(), B.copy())


def test_direct_diag_nstates_and_complex_sqrtm(monkeypatch, tmp_path):
    fake = _FakeTD()
    fake.chkfile = str(tmp_path / 'fake.chk')
    ab = _rhf_ab()
    fake.get_ab = lambda mf=None, frozen=None: ab

    orig = cvs_rhf.sqrtm

    def complex_sqrtm(m):
        return numpy.asarray(orig(m), dtype=complex)

    monkeypatch.setattr(cvs_rhf, 'sqrtm', complex_sqrtm)
    e, xy = cvs_rhf.direct_diag_rpa_kernel(fake, nstates=1)
    assert e.shape[0] == 1
    e, xy = cvs_rhf.direct_diag_tda_kernel(fake, nstates=1)
    assert e.shape[0] == 1

    fake_u = _FakeTD()
    fake_u.chkfile = str(tmp_path / 'fake_u.chk')
    uab = _uhf_ab()
    fake_u.get_ab = lambda mf=None, frozen=None: uab
    monkeypatch.setattr(cvs_uhf, 'sqrtm', complex_sqrtm)
    e, xy = cvs_uhf.direct_diag_rpa_kernel(fake_u, nstates=1)
    assert e.shape[0] == 1
    e, xy = cvs_uhf.direct_diag_tda_kernel(fake_u, nstates=1)
    assert e.shape[0] == 1

    fake_g = _FakeTD()
    fake_g.chkfile = str(tmp_path / 'fake_g.chk')
    fake_g.get_ab = lambda mf=None, frozen=None: ab
    monkeypatch.setattr(cvs_ghf, 'sqrtm', complex_sqrtm)
    e, xy = cvs_ghf.direct_diag_rpa_kernel(fake_g, nstates=1)
    assert e.shape[0] == 1
    e, xy = cvs_ghf.direct_diag_tda_kernel(fake_g, nstates=1)
    assert e.shape[0] == 1


def test_xy_norm_warn(monkeypatch):
    fake = _FakeTD()
    fake.get_ab = lambda mf=None, frozen=None: _rhf_ab()
    real_norm = pyscf.lib.norm
    state = {'n': 0}

    def inflated_y(x):
        state['n'] += 1
        n = real_norm(x)
        if state['n'] % 2 == 0:
            return n * 10
        return n

    monkeypatch.setattr(pyscf.lib, 'norm', inflated_y)
    cvs_rhf.direct_diag_rpa_kernel(fake, nstates=1)

    fake_u = _FakeTD()
    fake_u.get_ab = lambda mf=None, frozen=None: _uhf_ab()
    state['n'] = 0
    cvs_uhf.direct_diag_rpa_kernel(fake_u, nstates=1)

    fake_g = _FakeTD()
    fake_g.get_ab = lambda mf=None, frozen=None: _rhf_ab()
    state['n'] = 0
    cvs_ghf.direct_diag_rpa_kernel(fake_g, nstates=1)


def test_no_fxc_and_direct_diag_together():
    mf = _he('RKS', xc='PBE')
    td = TDA(mf).cvs(no_fxc=True, direct_diag=True)
    td.kernel(nstates=1)
    assert td.e is not None


def test_no_fxc_via_get_ab_helpers():
    mol = pyscf.M(atom='Be 0 0 0', basis='sto-3g', verbose=0)
    mf = pyscf.scf.RHF(mol).run()
    td = TDA(mf).cvs()
    td.frozen = [i for i in range(mf.mo_occ.size) if mf.mo_occ[i] == 2][1:]
    a, b = _get_ab(td, no_fxc=True)
    assert a.shape[0] == 1
    a2, b2 = _get_ab(td, no_fxc=False)
    assert a2.shape[0] == 1
    cvs_rhf._get_ab(td, no_fxc=True)
    cvs_rhf._get_ab(td, no_fxc=False)

    mf_u = _he('UHF')
    td_u = TDA(mf_u).cvs()
    occ_a = numpy.where(mf_u.mo_occ[0] != 0)[0]
    occ_b = numpy.where(mf_u.mo_occ[1] != 0)[0]
    td_u.frozen = (occ_a[1:].tolist(), occ_b[1:].tolist())
    au, bu = _get_ab(td_u, no_fxc=True)
    assert au[0].shape[0] == 1
    cvs_uhf._get_ab(td_u, no_fxc=True)
    cvs_uhf._get_ab(td_u, no_fxc=False)

    mf_g = _he('GHF')
    td_g = TDA(mf_g).cvs()
    occ = numpy.where(mf_g.mo_occ != 0)[0]
    td_g.frozen = occ[1:].tolist()
    ag, bg = _get_ab(td_g, no_fxc=True)
    assert ag.shape[0] == 1
    cvs_ghf._get_ab(td_g, no_fxc=True)
    cvs_ghf._get_ab(td_g, no_fxc=False)


def test_kernel_x0_passthrough():
    mf = _he('RHF')
    td = TDA(mf).cvs()
    x0 = td.get_init_guess(mf, nstates=1)
    td.kernel(x0=x0, nstates=1)
    assert td.e is not None


def test_no_fxc_forces_direct_diag():
    mf = _he('RKS', xc='PBE')
    td = TDA(mf).cvs(no_fxc=True)
    td.kernel(nstates=1)
    assert td.direct_diag is True
    assert td.e is not None


def test_vanilla_tda_untouched():
    mf = _he('RHF')
    td = TDA(mf)
    assert not isinstance(td, CVS)
    td.kernel(nstates=1)
    assert not isinstance(td, CVS)
    assert td.e is not None
