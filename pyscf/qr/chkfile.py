'''Checkpoint I/O for :class:`QR`.

QR checkpoints capture linear-response results *after* ``tdobj.kernel()`` and
*before* :meth:`QR.kernel` (QR stage).  Only manifold LR data are stored under
the ``qr/`` namespace.  ``mol``, ``mo_coeff``, and the mean-field object are
**not** written to disk; pass a live ``mf`` to :meth:`QR.from_chk`.
'''

from pyscf import lib

# needed for save_dip_mo
from pyscf.tdscf.rhf import _charge_center
import numpy

from pyscf.qr.manifold import Manifold


def _qr_key(name):
    return f'qr/{name}'


def save_qr(qrobj, chkfile=None):
    '''Write LR checkpoint to *chkfile*.

    Persists manifold JSON blobs only.  Intended to be called after linear
    response and before :meth:`QR.kernel`.
    '''
    chkfile = chkfile or qrobj.chkfile
    if not chkfile:
        return

    lib.chkfile.save(chkfile, _qr_key('manifold_n'), qrobj.manifold_n.dump())
    if qrobj.manifold_m is not qrobj.manifold_n:
        lib.chkfile.save(chkfile, _qr_key('manifold_m'),
                         qrobj.manifold_m.dump())


def load_manifold_n(chkfile, mf):
    '''Load the primary (N) manifold from a checkpoint file.'''
    data = lib.chkfile.load(chkfile, _qr_key('manifold_n'))
    return Manifold.loads(data, mf)


def load_manifold_m(chkfile, mf, manifold_n):
    '''Load the secondary (M) manifold, defaulting to *manifold_n*.'''
    try:
        data = lib.chkfile.load(chkfile, _qr_key('manifold_m'))
    except KeyError:
        return manifold_n
    if data is None:
        return manifold_n
    return Manifold.loads(data, mf)


def save_dipole_mo(chkfile, mf, dip_mo=None):
    '''Save the dipole moment integrals in the MO basis to `qr/dipole_mo`.'''
    if dip_mo is None:
        mol = mf.mol
        coeff = mf.mo_coeff
        with mol.with_common_orig(_charge_center(mol)):
            dip_ao = mol.intor_symmetric('int1e_r', comp=3)
        dip_mo = numpy.einsum('xpq,pr,qs->xrs',
                              dip_ao, coeff.conj(), coeff)

    lib.chkfile.save(chkfile, _qr_key('dipole_mo'), dip_mo)
    return dip_mo


def load_dipole_mo(chkfile):
    '''Load the dipole moment integrals in the MO basis from `qr/dipole_mo`.'''
    return lib.chkfile.load(chkfile, _qr_key('dipole_mo'))
