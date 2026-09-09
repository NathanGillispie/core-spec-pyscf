'''MP-ZORA geometry optimization: residual gradient at the optimized geometry.'''

import numpy as np
import pytest

pytest.importorskip('geometric')

from pyscf.zora.test.common import (
    GEOMOPT_GNORM_TOL, geomopt_gnorm, make_ghf, make_gks, make_hf_mol,
    make_rhf, make_rks, make_uhf, make_uks, requires_ghf_grad,
)


def test_rhf_geomopt():
    from pyscf.zora.test.common import CONV_PARAMS
    mol = make_hf_mol(1.3)
    mf = make_rhf(mol, conv_tol=1e-10)
    mol_eq = mf.Gradients().optimizer().kernel(CONV_PARAMS)
    mf_eq = make_rhf(mol_eq, conv_tol=1e-10)
    mf_eq.kernel()
    de = mf_eq.Gradients().kernel()
    assert np.linalg.norm(de) < GEOMOPT_GNORM_TOL


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


@requires_ghf_grad
def test_ghf_geomopt():
    gnorm, _, _ = geomopt_gnorm(lambda m: make_ghf(m, conv_tol=1e-10),
                                make_hf_mol(1.3))
    assert gnorm < GEOMOPT_GNORM_TOL, gnorm


@requires_ghf_grad
def test_gks_geomopt():
    gnorm, _, _ = geomopt_gnorm(lambda m: make_gks(m, conv_tol=1e-10),
                                make_hf_mol(1.3))
    assert gnorm < GEOMOPT_GNORM_TOL, gnorm
