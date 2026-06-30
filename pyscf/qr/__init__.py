'''
Quadratic response (QR) TDDFT for restricted references.

Typical workflow for excited-to-excited transition density matrices (2TDM)::

    from pyscf.tdscf import RPA
    import pyscf.qr

    mf = dft.RKS(mol).run()

    td_n = RPA(mf, frozen=core_idx).set(nstates=80)
    td_m = RPA(mf, frozen=other_core_idx).set(nstates=40)

    qr = pyscf.qr.QR(td_n, td_m)
    qr.save('qr.chk')          # LR checkpoint: call before qr.kernel()

    qr = pyscf.qr.QR.from_chk('qr.chk', mf)
    qr.kernel()                # QR stage (in-memory Gxc only)
    tdm = qr.get_2tdm(2, 0)

When both excited states come from the same active occupied subspace::

    qr = pyscf.qr.QR(td_n)
    tdm = qr.get_2tdm(1, 4)

TDSCF objects are consumed at initialization: linear response runs if needed,
manifolds are built, and the tdobjs are not retained on the QR object.

See ``AGENT.md`` for architecture notes and ``reference/on-the-fly/`` for
reference implementations.
'''

from pyscf.qr.hf import QR, qr_class_for_mf
from pyscf.qr.manifold import Manifold, gxc_tensor_shape
from pyscf.qr.rhf import RQR, LazyGxc, EagerGxc
from pyscf.qr.uhf import UQR
from pyscf.qr.ghf import GQR

QR = RQR

__all__ = [
    'Manifold',
    'QR',
    'qr_class_for_mf',
    'RQR',
    'UQR',
    'GQR',
    'LazyGxc',
    'EagerGxc',
    'gxc_tensor_shape',
]
