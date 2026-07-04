'''Test to make sure lazy and eager evaluation of Gxc produce the same result.'''

import numpy
import pytest
from pyscf import gto, lib, scf, dft
from pyscf.tdscf import RPA, TDA

from pyscf.qr import QR
from math import isclose


@pytest.fixture(scope='module')
def lih_td():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='PBE0')
    mf.grids.level = 1
    mf.kernel()
    return RPA(mf).run(nstates=4)

@pytest.fixture
def lih_eager(lih_td, pair):
    qrobj = QR(lih_td, precompute_gxc=True).kernel()
    tdm = qrobj.get_2tdm(*pair)
    tdip = qrobj.transition_dipole(tdm)
    return float(numpy.linalg.norm(tdip))

@pytest.fixture
def lih_lazy(lih_td, pair):
    qrobj = QR(lih_td, precompute_gxc=False)
    tdm = qrobj.get_2tdm(*pair)
    tdip = qrobj.transition_dipole(tdm)
    return float(numpy.linalg.norm(tdip))

@pytest.mark.parametrize(
    'pair',
     [(0, 1), (0, 3), (1, 3)],
     ids = ['0-1', '0-3', '1-3']
)
def test_lih(lih_eager, lih_lazy):
    assert isclose(lih_eager, lih_lazy, abs_tol=1e-7)

