'''MP-ZORA geometry optimization: residual gradient at the optimized geometry.'''

import pytest

pytest.importorskip('geometric')

from pyscf.zora.test.common import (
    GEOMOPT_GNORM_TOL, geomopt_gnorm, make_ghf, make_gks, make_hf_mol,
    make_rhf, make_rks, make_uhf, make_uks,
)


def test_rhf_geomopt():
    gnorm, _, _ = geomopt_gnorm(lambda m: make_rhf(m, conv_tol=1e-10),
                                make_hf_mol(1.3))
    assert gnorm < GEOMOPT_GNORM_TOL, gnorm


def test_rks_geomopt():
    gnorm, _, _ = geomopt_gnorm(lambda m: make_rks(m, conv_tol=1e-10),
                                make_hf_mol(1.3))
    assert gnorm < GEOMOPT_GNORM_TOL, gnorm


def test_uhf_geomopt():
    gnorm, _, _ = geomopt_gnorm(lambda m: make_uhf(m, conv_tol=1e-10),
                                make_hf_mol(1.3))
    assert gnorm < GEOMOPT_GNORM_TOL, gnorm


def test_uks_geomopt():
    gnorm, _, _ = geomopt_gnorm(lambda m: make_uks(m, conv_tol=1e-10),
                                make_hf_mol(1.3))
    assert gnorm < GEOMOPT_GNORM_TOL, gnorm


def test_ghf_geomopt():
    gnorm, _, _ = geomopt_gnorm(lambda m: make_ghf(m, conv_tol=1e-10),
                                make_hf_mol(1.3))
    assert gnorm < GEOMOPT_GNORM_TOL, gnorm


def test_gks_geomopt():
    gnorm, _, _ = geomopt_gnorm(lambda m: make_gks(m, conv_tol=1e-10),
                                make_hf_mol(1.3))
    assert gnorm < GEOMOPT_GNORM_TOL, gnorm
