'''
The purpose of this test is to test the syntax of core_idx.
The option can be passed in with both the tdobj attribute and kernel keyword argument.
'''
import pyscf
import pyscf.cvs
from pyscf.tdscf import RPA, TDA

import pytest

testdata = [
    ("RKS", [0]),
    ("UKS", ([0], [0])),
    ("GKS", [0]),
    ("RHF", [0]),
    ("UHF", ([0], [0])),
    ("GHF", [0])
]

@pytest.mark.parametrize("ref,core_idx", testdata)
def test_tda(ref,core_idx):
    mol = pyscf.M(atom='H 0 0 0; H .5 0 0', basis='sto-3g')
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()
    tdobj = TDA(mf)
    tdobj.core_idx = core_idx
    tdobj.kernel()

    del tdobj.core_idx
    tdobj.core_valence(core_idx=core_idx)
    tdobj.kernel()

@pytest.mark.parametrize("ref,core_idx", testdata)
def test_rpa(ref,core_idx):
    mol = pyscf.M(atom='H 0 0 0; H .5 0 0', basis='sto-3g')
    mf = eval(f'pyscf.scf.{ref}(mol)')
    mf.kernel()
    tdobj = RPA(mf)
    tdobj.core_idx = core_idx
    tdobj.kernel()

    del tdobj.core_idx
    tdobj.core_valence(core_idx=core_idx)
    tdobj.kernel()
