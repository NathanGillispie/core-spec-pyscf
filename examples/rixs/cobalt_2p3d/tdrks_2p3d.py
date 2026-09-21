#!/usr/bin/env python

from pyscf import gto, dft, tdscf
from pyscf.tools import molden
import numpy
numpy.set_printoptions(suppress=True, linewidth=160, precision=4)

# modules from core-spec-pyscf
import pyscf.zora
import pyscf.cvs
import pyscf.qr
import pyscf.rixs

NAME = 'tdrks_2p3d'
MANIFOLD1 = range(2,5) # Co 2p
MANIFOLD2 = range(9,14) # Co 3d
QR_CHECKPOINT = NAME + '_qr.chk'
XAS_CSV = 'sticks_' + NAME + '.csv'


def get_mol():
    basis = {
       'Co': gto.basis.load('def2-TZVP', 'Co'),
    }

    return gto.M(
        atom = 'Co 0 0 0',
        basis=basis,
        charge = -3,
        spin = 0,
        verbose = 4,
    )


def get_mf():
    """Set up a ZORA energy calculation."""
    mol = get_mol()
    return dft.RKS(mol, xc='PBE0').set(max_cycle=200).zora()


def _guess_dm(mf):
    """If has guess from optimize, use it. Returns a density matrix."""
    import os
    if os.path.exists('optimize.chk'):
        d = {'key': 'chkfile', 'chkfile':'optimize.chk'}
    else:
        d = {'key': 'atom'}
    return mf.get_init_guess(**d)


def kernel(mf=None, stability_iter=3):
    """Guess orbitals, run kernel, (stability), analyze, save molden.

    Try to load the checkfile if it exists. Then kernel.
    stability_iter can be set to 0 to turn off stability analysis.
    """
    mf = mf or get_mf()
    dm = _guess_dm(mf)
    e_tot = mf.kernel(dm)
    # Ensure the molecular orbitals are stable.
    for i in range(stability_iter):
        print(f'   stability analysis: iter {i+1}')
        C, _, stable, _ = mf.stability(return_status=True)
        if stable: break
        dm = mf.make_rdm1(mo_coeff=C)
        e_tot = mf.kernel(dm)
    # Print out some information for me.
    mf.analyze()
    molden.from_scf(mf, NAME+'.molden')
    return mf


def run_linear_response(mf):
    """Linear response.

    Use core-spec-pyscf's direct diagonalization and CVS support. Request all
    TDA roots so the QR checkpoint contains the same manifolds as the old
    explicit eigensolver.
    """

    td_n = tdscf.TDA(mf).cvs(core_idx=MANIFOLD1, direct_diag=True)
    td_n.kernel()
    print(f'Saving intermediate XAS sticks: {XAS_CSV}', flush=True)
    numpy.savetxt(
        XAS_CSV,
        numpy.column_stack((td_n.e, numpy.abs(td_n.oscillator_strength()))),
        delimiter=',',
    )

    td_f = tdscf.TDA(mf).cvs(core_idx=MANIFOLD2, direct_diag=True)
    td_f.kernel()

    # RIXS owns the combined SCF, QR, ZORA, and dipole checkpoint.
    qr = pyscf.qr.QR(td_n, td_f, chkfile=QR_CHECKPOINT)
    rixs = pyscf.rixs.RIXS(mf, qr, chkfile=QR_CHECKPOINT)

    rixs.save()
    print(f'Saved QR checkpoint: {QR_CHECKPOINT}', flush=True)
    return rixs


if __name__ == '__main__':
    mf = get_mf()
    kernel(mf)
    run_linear_response(mf)
