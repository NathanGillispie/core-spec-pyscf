'''RIXS calculation driver using the Kramers-Heisenberg equation.

Stores QR and mean-field references and provides RIXS-specific response
quantities built from them.
'''

from pyscf import lib

from pyscf.qr.hf import QR as QRBase

from pyscf.rixs import chkfile as _chkfile
from pyscf.rixs.response import (
    ground_transition_dipoles as _ground_transition_dipoles,
)


class RIXS(lib.StreamObject):
    '''Container for a mean-field reference and a QR calculation.'''

    _keys = {
        'mf',
        'qr',
        'mol',
        'chkfile',
        'dipole_mo',
    }

    def __init__(self, mf, qr, *, chkfile=None):
        if not isinstance(qr, QRBase):
            raise TypeError(
                f'qr must be a QR driver, got {type(qr).__name__}')
        if qr.mf is not mf:
            raise ValueError('mf and qr must use the same mean-field object')

        self.mf = mf
        self.qr = qr
        self.mol = mf.mol
        self.chkfile = chkfile if chkfile is not None else qr.chkfile
        self.dipole_mo = None

    def save(self, chkfile=None):
        '''Write this RIXS calculation to a checkpoint file.'''
        _chkfile.save_rixs(self, chkfile=chkfile)
        return self

    def ground_transition_dipoles(self, states=None):
        '''Return ground-to-intermediate transition dipoles.

        Parameters
        ----------
        states : array_like of int, optional
            0-based indices into ``qr.manifold_n``.  Defaults to all
            intermediate states.

        Returns
        -------
        ndarray
            Transition dipoles with shape ``(3, len(states))``.
        '''
        return _ground_transition_dipoles(
            self.mol,
            self.qr.mo_coeff,
            self.qr.mo_occ,
            self.qr.manifold_n,
            states,
        )

    @classmethod
    def from_chk(cls, chkfile, mf, *, precompute_gxc=False):
        '''Restore a RIXS calculation into a live MF template.'''
        return _chkfile.load_rixs(
            chkfile,
            mf,
            precompute_gxc=precompute_gxc,
        )
