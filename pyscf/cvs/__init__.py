'''
Core-valence separation for TDDFT calculations.

Wrap a TDSCF object with :meth:`cvs` (assign the return value)::

    import pyscf.cvs
    td = TDA(mf).cvs(core_idx=[0, 1, 2])
    td.kernel()

``core_idx`` is converted to PySCF's ``frozen`` attribute (valence occupied
orbitals are frozen). The SCF ``mo_coeff`` / ``mo_occ`` / ``mo_energy`` arrays
and ``mol.nelec`` are left unchanged.

Select occupied core orbitals by an inclusive MO-energy window::

    td = TDA(mf).cvs(core_window=(-20.0, -10.0))

For UHF/UKS, one window is applied to both spins, or separate alpha and beta
windows can be supplied::

    td = TDA(mf).cvs(core_window=((-20.0, -10.0), (-19.0, -9.0)))

For UHF/UKS, pass a ``(alpha, beta)`` pair::

    td = TDA(mf).cvs(core_idx=([0, 1], [0, 1]))

Direct diagonalization and dropping :math:`f_\\text{xc}`::

    td = TDA(mf).cvs(direct_diag=True, no_fxc=True)
    td.kernel()
'''

from inspect import signature

from pyscf.tdscf.rhf import TDBase

if 'frozen' not in signature(TDBase.__init__).parameters:
    raise ImportError(
        'pyscf.cvs requires PySCF TDSCF frozen-orbital support '
        '(PySCF >= 2.10). Install core-spec-pyscf[cvs].')

from pyscf.cvs.td import CVS, cvs

TDBase.cvs = cvs
