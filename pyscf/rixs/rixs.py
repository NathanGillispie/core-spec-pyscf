'''RIXS calculation driver using the Krammers-Heisenberg equation.

Currently stores QR and mf references. In the future, I will add the
ability to specify energy ranges, produce the ground-to-intermediate and
intermediate-to-final transition moments. For now, I'm adding features
I know I will need for my current workflow.
'''

from pyscf import lib

from pyscf.qr.hf import QR as QRBase

from pyscf.rixs import chkfile as _chkfile


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

    @classmethod
    def from_chk(cls, chkfile, mf, *, precompute_gxc=False):
        '''Restore a RIXS calculation into a live MF template.'''
        return _chkfile.load_rixs(
            chkfile,
            mf,
            precompute_gxc=precompute_gxc,
        )
