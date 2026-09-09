'''Singlet–triplet SO coupling utility (restricted TD amplitudes).'''

import numpy as np
import pyscf
import pyscf.zora
from pyscf.tdscf import TDA
from pyscf.zora import compute_SO_coupling_matrix
from pyscf.zora.test.common import GRAD_GRID_LEVEL


def test_so_coupling_matrix_h2():
    mol = pyscf.gto.M(
        atom='H 0 0 0; H 0 0 0.74',
        basis='sto-3g',
        unit='Angstrom',
        verbose=0,
    )
    mf = pyscf.scf.RHF(mol).zora(grid_level=GRAD_GRID_LEVEL)
    mf.with_zora.spin_orbit = True
    mf.kernel()
    assert mf.with_zora.get_hso() is not None

    td_s = TDA(mf).set(nstates=1, singlet=True)
    td_s.verbose = 0
    td_s.kernel()
    td_t = TDA(mf).set(nstates=1, singlet=False)
    td_t.verbose = 0
    td_t.kernel()

    X_s = np.array([x for x, y in td_s.xy])
    X_t = np.array([x for x, y in td_t.xy])
    X_so, w_so = compute_SO_coupling_matrix(mf, X_s, td_s.e, X_t, td_t.e)
    assert X_so.shape == (1, 1)
    assert w_so.shape == (1, )
    assert np.isfinite(w_so).all()
    assert np.isfinite(X_so).all()
