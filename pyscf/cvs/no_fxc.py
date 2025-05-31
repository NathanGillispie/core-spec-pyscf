import numpy
from pyscf import lib, ao2mo
import pyscf.tdscf

@lib.with_doc(pyscf.tdscf.rhf.get_ab.__doc__)
def get_ab_no_fxc_rhf(mf, mo_energy=None, mo_coeff=None, mo_occ=None, add_hf=True):
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    # assert (mo_coeff.dtype == numpy.double)

    assert mo_coeff.dtype == numpy.float64
    mol = mf.mol
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = numpy.hstack((orbo,orbv))

    e_ia = mo_energy[viridx] - mo_energy[occidx,None]
    a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = numpy.zeros_like(a)

    if add_hf:
        eri_mo = ao2mo.general(mol, [orbo,mo,mo,mo], compact=False)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        a += numpy.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc]) * 2
        a -= numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:])

        b += numpy.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * 2
        b -= numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:])

    return a, b

@lib.with_doc(pyscf.tdscf.uhf.get_ab.__doc__)
def get_ab_no_fxc_uhf(mf, mo_energy=None, mo_coeff=None, mo_occ=None, add_hf=True):
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ

    assert mo_coeff[0].dtype == numpy.float64
    mol = mf.mol
    nao = mol.nao_nr()
    occidx_a = numpy.where(mo_occ[0]==1)[0]
    viridx_a = numpy.where(mo_occ[0]==0)[0]
    occidx_b = numpy.where(mo_occ[1]==1)[0]
    viridx_b = numpy.where(mo_occ[1]==0)[0]
    orbo_a = mo_coeff[0][:,occidx_a]
    orbv_a = mo_coeff[0][:,viridx_a]
    orbo_b = mo_coeff[1][:,occidx_b]
    orbv_b = mo_coeff[1][:,viridx_b]
    nocc_a = orbo_a.shape[1]
    nvir_a = orbv_a.shape[1]
    nocc_b = orbo_b.shape[1]
    nvir_b = orbv_b.shape[1]
    mo_a = numpy.hstack((orbo_a,orbv_a))
    mo_b = numpy.hstack((orbo_b,orbv_b))
    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b

    e_ia_a = (mo_energy[0][viridx_a,None] - mo_energy[0][occidx_a]).T
    e_ia_b = (mo_energy[1][viridx_b,None] - mo_energy[1][occidx_b]).T
    a_aa = numpy.diag(e_ia_a.ravel()).reshape(nocc_a,nvir_a,nocc_a,nvir_a)
    a_bb = numpy.diag(e_ia_b.ravel()).reshape(nocc_b,nvir_b,nocc_b,nvir_b)
    a_ab = numpy.zeros((nocc_a,nvir_a,nocc_b,nvir_b))
    b_aa = numpy.zeros_like(a_aa)
    b_ab = numpy.zeros_like(a_ab)
    b_bb = numpy.zeros_like(a_bb)
    a = (a_aa, a_ab, a_bb)
    b = (b_aa, b_ab, b_bb)

    # Add exact hartree exchange
    if add_hf:
        eri_aa = ao2mo.general(mol, [orbo_a,mo_a,mo_a,mo_a], compact=False)
        eri_ab = ao2mo.general(mol, [orbo_a,mo_a,mo_b,mo_b], compact=False)
        eri_bb = ao2mo.general(mol, [orbo_b,mo_b,mo_b,mo_b], compact=False)
        eri_aa = eri_aa.reshape(nocc_a,nmo_a,nmo_a,nmo_a)
        eri_ab = eri_ab.reshape(nocc_a,nmo_a,nmo_b,nmo_b)
        eri_bb = eri_bb.reshape(nocc_b,nmo_b,nmo_b,nmo_b)
        a_aa, a_ab, a_bb = a
        b_aa, b_ab, b_bb = b

        a_aa += numpy.einsum('iabj->iajb', eri_aa[:nocc_a,nocc_a:,nocc_a:,:nocc_a])
        a_aa -= numpy.einsum('ijba->iajb', eri_aa[:nocc_a,:nocc_a,nocc_a:,nocc_a:])
        b_aa += numpy.einsum('iajb->iajb', eri_aa[:nocc_a,nocc_a:,:nocc_a,nocc_a:])
        b_aa -= numpy.einsum('jaib->iajb', eri_aa[:nocc_a,nocc_a:,:nocc_a,nocc_a:])

        a_bb += numpy.einsum('iabj->iajb', eri_bb[:nocc_b,nocc_b:,nocc_b:,:nocc_b])
        a_bb -= numpy.einsum('ijba->iajb', eri_bb[:nocc_b,:nocc_b,nocc_b:,nocc_b:])
        b_bb += numpy.einsum('iajb->iajb', eri_bb[:nocc_b,nocc_b:,:nocc_b,nocc_b:])
        b_bb -= numpy.einsum('jaib->iajb', eri_bb[:nocc_b,nocc_b:,:nocc_b,nocc_b:])

        a_ab += numpy.einsum('iabj->iajb', eri_ab[:nocc_a,nocc_a:,nocc_b:,:nocc_b])
        b_ab += numpy.einsum('iajb->iajb', eri_ab[:nocc_a,nocc_a:,:nocc_b,nocc_b:])

    return a, b


@lib.with_doc(pyscf.tdscf.ghf.get_ab.__doc__)
def get_ab_no_fxc_ghf(mf, mo_energy=None, mo_coeff=None, mo_occ=None, add_hf=True):
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ

    mol = mf.mol
    nmo = mo_occ.size
    nao = mol.nao
    occidx = numpy.where(mo_occ==1)[0]
    viridx = numpy.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = numpy.hstack((orbo,orbv))
    moa = mo[:nao].copy()
    mob = mo[nao:].copy()
    orboa = orbo[:nao]
    orbob = orbo[nao:]
    nmo = nocc + nvir

    e_ia = mo_energy[viridx] - mo_energy[occidx,None]
    a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir).astype(mo_coeff.dtype)
    b = numpy.zeros_like(a)

    if add_hf:
        if mo_coeff.dtype == numpy.double:
            eri_mo  = ao2mo.general(mol, [orboa,moa,moa,moa], compact=False)
            eri_mo += ao2mo.general(mol, [orbob,mob,mob,mob], compact=False)
            eri_mo += ao2mo.general(mol, [orboa,moa,mob,mob], compact=False)
            eri_mo += ao2mo.general(mol, [orbob,mob,moa,moa], compact=False)
            eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        else:
            eri_ao = mol.intor('int2e').reshape([nao]*4)
            eri_mo_a = lib.einsum('pqrs,pi,qj->ijrs', eri_ao, orboa.conj(), moa)
            eri_mo_a+= lib.einsum('pqrs,pi,qj->ijrs', eri_ao, orbob.conj(), mob)
            eri_mo = lib.einsum('ijrs,rk,sl->ijkl', eri_mo_a, moa.conj(), moa)
            eri_mo+= lib.einsum('ijrs,rk,sl->ijkl', eri_mo_a, mob.conj(), mob)
        a += numpy.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc].conj())
        a -= numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:].conj())
        b += numpy.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:].conj())
        b -= numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:].conj())

    return a, b
