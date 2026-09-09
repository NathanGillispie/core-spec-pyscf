'''
Core-valence separation for TDDFT calculations.

Wrap a TDSCF object with :meth:`cvs` (assign the return value)::

    import pyscf.cvs
    td = TDA(mf).cvs(core_idx=[0, 1, 2])
    td.kernel()

``core_idx`` is converted to PySCF's ``frozen`` attribute (valence occupied
orbitals are frozen). The SCF ``mo_coeff`` / ``mo_occ`` / ``mo_energy`` arrays
and ``mol.nelec`` are left unchanged.

For UHF/UKS, pass a ``(alpha, beta)`` pair::

    td = TDA(mf).cvs(core_idx=([0, 1], [0, 1]))

Direct diagonalization and dropping :math:`f_\\text{xc}`::

    td = TDA(mf).cvs(direct_diag=True, no_fxc=True)
    td.kernel()
'''

from pyscf.tdscf.rhf import TDBase
from pyscf.cvs.td import CVS, cvs

TDBase.cvs = cvs
