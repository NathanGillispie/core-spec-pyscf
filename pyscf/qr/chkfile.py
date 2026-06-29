'''Checkpoint I/O for :class:`QR`.

QR checkpoints capture linear-response results *after* ``tdobj.kernel()`` and
*before* :meth:`QR.kernel` (QR stage).  Only manifold LR data are stored under
the ``qr/`` namespace.  ``mol``, ``mo_coeff``, and the mean-field object are
**not** written to disk; pass a live ``mf`` to :meth:`QR.from_chk`.
'''

from pyscf import lib

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
        lib.chkfile.save(chkfile, _qr_key('manifold_m'), qrobj.manifold_m.dump())


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
