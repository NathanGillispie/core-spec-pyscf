'''Dipole-integral helpers for quadratic-response calculations.'''

import numpy

from pyscf.tdscf.rhf import _charge_center


def compute_dipole_mo(mol, mo_coeff):
    '''Return length-gauge electric-dipole integrals in the MO basis.'''
    with mol.with_common_orig(_charge_center(mol)):
        dip_ao = mol.intor_symmetric('int1e_r', comp=3)
    return numpy.einsum(
        'xpq,pr,qs->xrs',
        dip_ao,
        mo_coeff.conj(),
        mo_coeff,
    )
