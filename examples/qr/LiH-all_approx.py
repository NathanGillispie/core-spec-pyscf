#!/usr/bin/env python
'''
Replicating https://doi.org/10.1063/1.4963749/
Parker, S. M.; Roy, S.; Furche, F. Unphysical Divergences in Response Theory. J. Chem. Phys. 2016, 145 (13), 134105.
Section III. B. LiH

Comparing excitation energies + transition dipole moments
RKS (PBE0 + def2-SVP) + TDDFT
FCI (singlet def2-SVP)
'''

from functools import reduce
from pathlib import Path

import numpy
import pyscf.dft
import pyscf.fci
import pyscf.tdscf
import pyscf.qr
from pyscf.qr import QR

numpy.set_printoptions(suppress=True, precision=6, linewidth=240)
au2ev = pyscf.data.nist.HARTREE2EV

bond_lengths = numpy.arange(2, 3.01, .01) #Å

# Excitation frequencies
w01 = []
w14 = []
fci_w01 = []
fci_w14 = []
# Transition dipole moments
tdip14 = []
fci_tdip14 = []
pw_tdip14 = []
g0_tdip14 = []

rks_scanner = pyscf.gto.M(verbose=0).apply(pyscf.dft.RKS, xc='PBE0').as_scanner()
rks_scanner.grids.level = 5
for BOND_LENGTH in bond_lengths:
    mol = pyscf.gto.M(
        atom=f'Li 0 0 0; H 0 0 {BOND_LENGTH}',
        basis='def2-SVP',
        symmetry=False
    )
    e0 = rks_scanner(mol)
    print(f'Bond length: {BOND_LENGTH:.2f}')

    ## FCI part
    cisolver = pyscf.fci.FCI(rks_scanner, singlet=True).set(nroots=5)
    (e0,e1,_,_,e4), (v0,v1,_,_,v4) = cisolver.kernel()
    fci_w01.append((e1-e0)*au2ev)
    fci_w14.append((e4-e1)*au2ev)
    # print(f'  FCI  ω  (0->1) {fci_w01[-1]:7.6f}    (1->4) {fci_w14[-1]:7.6f}')

    charges = mol.atom_charges()
    coords = mol.atom_coords()  # in a.u.
    nuc_charge_center = numpy.einsum('z,zx->x', charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)
    dip = mol.intor('cint1e_r_sph', comp=3)

    orbs = rks_scanner.mo_coeff
    norb = orbs.shape[1]
    nelec = (2,2)
    t_dm1 = cisolver.trans_rdm1(v1, v4, norb, nelec)
    # transform density matrix to AO representation
    t_dm1_ao = reduce(numpy.dot, (orbs, t_dm1, orbs.T))
    d14 = numpy.einsum('xij,ji->x', dip, t_dm1_ao)
    fci_tdip14.append((d14.dot(d14))**.5)
    print(f'  FCI  μ14 {fci_tdip14[-1]:.6f}')


    ## TDDFT quadratic response part
    tdobj = pyscf.tdscf.RPA(rks_scanner).set(nroots=4)
    e1, _, _, e4 = tdobj.kernel()[0]

    qrobj = QR(tdobj)
    tdm14 = qrobj.get_2tdm(0, 3)
    tdip = qrobj.transition_dipole(tdm14)
    tdip = abs(tdip.dot(tdip)**.5)

    w01.append((e1)*au2ev)
    w14.append((e4-e1)*au2ev)

    tdip14.append(tdip)

    # Pseudo-Wavefunction approximation
    qrobj.approximation = 'Pseudo'
    tdm14 = qrobj.get_2tdm(0, 3)
    tdip = qrobj.transition_dipole(tdm14)
    tdip = abs(tdip.dot(tdip)**.5)
    pw_tdip14.append(tdip)

    # Xab Yab == 0 app.
    qrobj.approximation = 'Nascimento'
    tdm14 = qrobj.get_2tdm(0, 3)
    tdip = qrobj.transition_dipole(tdm14)
    tdip = abs(tdip.dot(tdip)**.5)
    g0_tdip14.append(tdip)

    # print(f'RKS  ω  (0->1) {w01[-1]:7.5f}    (1->4) {w14[-1]:7.5f}')
    print(f'  RKS  μ14 {tdip14[-1]:.6f}')
    print(f'  PW   μ14 {pw_tdip14[-1]:.6f}')
    print(f'  G=0  μ14 {g0_tdip14[-1]:.6f}')

outfile = Path(__file__).with_suffix('.npz')
numpy.savez(
    outfile,
    bond_lengths=bond_lengths,
    w01=numpy.asarray(w01),
    w14=numpy.asarray(w14),
    fci_w01=numpy.asarray(fci_w01),
    fci_w14=numpy.asarray(fci_w14),
    fci_tdip14=numpy.asarray(fci_tdip14),
    tdip14=numpy.asarray(tdip14),
    pw_tdip14=numpy.asarray(pw_tdip14),
    g0_tdip14=numpy.asarray(g0_tdip14),
)
print(f'Saved data to {outfile}')
