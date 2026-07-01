'''
Smoke tests to help remind me what options I need to implement. Some combinations
are expected to fail due to NotImplementedErrors.
'''

import pytest
import numpy

from pyscf import gto, dft
from pyscf.tdscf import RPA, TDA
from pyscf.qr import QR

@pytest.fixture(scope="module")
def lih_mf():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='def2-SVP', verbose=0, symmetry=False)
    return dft.RKS(mol, xc='PBE0').run()

@pytest.mark.parametrize('td', [RPA, TDA])
@pytest.mark.parametrize('frozen_idx',
    [numpy.asarray([]), numpy.asarray([0])],
    ids = ["Freeze[]", "Freeze[0]"]
)
@pytest.mark.parametrize('precompute_gxc', [True, False], ids = ['EagerGxc', 'LazyGxc'])
def test_qr_options_smoke(lih_mf, precompute_gxc, td, frozen_idx):
    tdobj = td(lih_mf).run(nstates=4, frozen=frozen_idx)
    qrobj = QR(tdobj, precompute_gxc=precompute_gxc).kernel()
    qrobj.get_2tdm(0,3)

