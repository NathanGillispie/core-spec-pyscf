
import numpy
import scipy
from pyscf import dft, scf, lib
from pyscf.data.nist import LIGHT_SPEED
from pyscf.data.elements import ELEMENTS

from .modbas2c import modbas

'''
Zeroth-Order Regular Approximation (ZORA) for HF/KS objects

Typical usage:

>>> import pyscf.cvs
>>> mol = pyscf.gto.M(...)
>>> mf = pyscf.scf.GHF(mol).zora()
>>> mf.kernel()
'''

__author__ = 'Nathan Gillispie'

def get_zora_hcore(mf):
    '''
    Generates the ZORA scalar-relativistic kinetic integral.
    Replaces the get_hcore function of the SCF object
    '''

    if hasattr(mf, '_zora_hcore'):
        return mf._zora_hcore

    mol = mf.mol

    # Treutler + no pruning decreases numerical error
    grid = dft.gen_grid.Grids(mol)
    grid.prune = None
    grid.level = 8
    grid.build(with_non0tab=False)

    Z = mol.atom_charges()
    # Reads the model potential basis `modbas.2c` and returns the contraction coefficients
    # and square rooted exponents of the given atoms.
    c_a = []
    for z in Z:
        c_a.append(numpy.asarray(modbas[z]))

    # Computing effective potential and ZORA integration kernel
    veff = numpy.zeros(grid.coords.shape[0])
    for coords, (c, a), Z in zip(mol.atom_coords(), c_a, Z):
        PA = grid.coords - coords
        RPA = numpy.sqrt(numpy.sum(PA**2, axis=1))
        outer = numpy.outer(a, RPA)
        veff += numpy.einsum('i,i,ip->p', c, a, scipy.special.erf(outer)/outer, optimize=True)
        veff -= Z/RPA
    kernel = LIGHT_SPEED**2 / (2 * LIGHT_SPEED**2 - veff)

    # block_loop has a good memory estimate and cuts down on code
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    # but we have to do this cursed line
    grid.weights *= kernel

    T = numpy.zeros((mol.nao, mol.nao))
    for ao, _, weights, _ in \
            dft.numint.NumInt().block_loop(mol, grid, deriv=1, max_memory=max_memory):
        T += numpy.einsum('xip,xiq,i->pq', ao[1:], ao[1:], weights, optimize=True)

    mf._zora_hcore = T + mol.intor('int1e_nuc')
    mf.get_hcore = lambda self: mf._zora_hcore

    return mf

def zora_hcore_ghf(mf):
    mf = get_zora_hcore(mf)
    mf._zora_hcore = scipy.linalg.block_diag(mf._zora_hcore, mf._zora_hcore)
    return mf

scf.rhf.RHF.zora = lambda self: get_zora_hcore(self)
scf.uhf.UHF.zora = lambda self: get_zora_hcore(self)
scf.ghf.GHF.zora = lambda self: zora_hcore_ghf(self)

dft.rks.RKS.zora = lambda self: get_zora_hcore(self)
dft.uks.UKS.zora = lambda self: get_zora_hcore(self)
dft.gks.GKS.zora = lambda self: zora_hcore_ghf(self)

