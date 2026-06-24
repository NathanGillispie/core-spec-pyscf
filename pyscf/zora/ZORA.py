
import numpy
import scipy
from pyscf import dft, scf, lib
from pyscf.data.nist import LIGHT_SPEED
from pyscf.data.elements import ELEMENTS

from .modbas2c import modbas

"""
Zeroth-Order Regular Approximation (ZORA) for HF/KS objects

Typical usage:

>>> import pyscf.cvs
>>> mol = pyscf.gto.M(...)
>>> mf = pyscf.scf.GHF(mol).zora()
>>> mf.kernel()
"""

__author__ = "Nathan Gillispie"

def _block_loop(mol, grids, kernel, deriv=0, max_memory=2000):
    """
    Define this macro to loop over grids by blocks.
    """
    ngrids = grids.coords.shape[0]
    nao = mol.nao
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    assert kernel.shape[0] == ngrids

    from pyscf.dft.gen_grid import BLKSIZE, NBINS, ALIGNMENT_UNIT
    blksize = int(max_memory*1e6/((comp+1)*nao*8*BLKSIZE))
    blksize = max(4, min(blksize, ngrids//BLKSIZE+1, 1200)) * BLKSIZE
    assert blksize % BLKSIZE == 0

    buf = dft.numint._empty_aligned(comp * blksize * nao)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[ip0:ip1]
        weight = grids.weights[ip0:ip1]
        _kernel = kernel[ip0:ip1]
        ao = dft.numint.eval_ao(mol, coords, deriv=deriv, cutoff=grids.cutoff, out=buf)
        yield ao, weight, _kernel

def _hf_zora_get_hcore(self, mol=None):
    """Core Hamiltonian"""
    if mol is not None:
        if mol is not self.mol:
            lib.logger.warn(self, "ZORA mf.get_hcore called with mol != mf.mol. "
                                  "Not recomputing ZORA integrals.")
    if not hasattr(self, "_zora_hcore"):
        lib.logger.warn(self, "Did you delete the mf._zora_hcore attribute?")
    return self._zora_hcore

def add_zora_hcore_attribute_(mf, mol=None, grid=None, max_memory=None):
    """Patches mf.get_hcore() to return ZORA integrals.

    Computes the ZORA relativistic integral(s) and caches scalar part to the
    self._zora_hcore attribute. Computes spin-orbit integrals if using
    generalized refernce and mf._spin_orbit == True.

    Parameters
    ----------
    mf : scf.hf.SCF instance
        The SCF reference object to be modified.
    mol : gto.mol.Mole instance, default=None
    grid : dft.numint.Grid instance, default=None
        ZORA integration grid. Can overwrite grid options.
    max_memory: int | float, default=mf.max_memory
        Caps block size for ZORA integration.

    Returns
    -------
    mf : scf.hf.SCF instance

    Examples
    --------
    >>> import pyscf
    >>> import pyscf.zora # This plugin
    ...
    >>> mf = pyscf.scf.RKS(mol)
    >>> mf.zora() # This method
    >>> mf.kernel()
    """

    if hasattr(mf, "_zora_hcore"):
        lib.logger.warn(mf, "mf object already has _zora_hcore attribute.")
        return mf

    if mol is None: mol = mf.mol
    if max_memory is None: max_memory = mf.max_memory
    do_spin_orbit = getattr(mf, "_spin_orbit", False)
    ghf_method = any("GHF" == t.__name__ for t in mf.__class__.__mro__)

    if do_spin_orbit and not ghf_method:
        lib.logger.warn(mf, "Not using GHF method. Spin-orbit coupling will not work! "
                            "Set mf._spin_orbit = False.")
        return mf

    zora_hcore, eps_scal_ao, spin_orbit_hamiltonian = _compute_zora(mol, grid, max_memory, do_spin_orbit)

    mf._eps_scal_ao = eps_scal_ao
    if ghf_method:
        mf._zora_hcore = scipy.linalg.block_diag(zora_hcore, zora_hcore)
        if do_spin_orbit:
            mf._zora_hcore = mf._zora_hcore.astype(complex) + spin_orbit_hamiltonian * 1j
            assert numpy.allclose(mf._zora_hcore, mf._zora_hcore.conj().T)
    else:
        mf._zora_hcore = zora_hcore

    from types import MethodType
    mf.get_hcore = MethodType(_hf_zora_get_hcore, mf)

def _compute_zora(mol, grid=None, max_memory=2000, do_spin_orbit=False):
    """
    Computes the ZORA relativistic kinetic integral (scalar or spin-orbit).
        Returns scalar-integral, ecp_ao, SO-integral | None
    ecp_ao is used to correct molecular orbital energies after SCF.
    """

    # Treutler + no pruning decreases numerical error
    if grid is None:
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
    for atom_coord, (c, a), Z in zip(mol.atom_coords(), c_a, Z):
        PA = grid.coords - atom_coord
        # Distance of each coord to atom
        RPA = numpy.sqrt(numpy.sum(PA**2, axis=1))
        outer = numpy.outer(a, RPA)
        # Integrating the 1e relativistic correction term.
        veff += numpy.einsum('i,i,ip->p', c, a, scipy.special.erf(outer)/outer, optimize=True)
        # Include non-relativistic 1e nuclear attraction.
        veff -= Z/RPA

    C2 = LIGHT_SPEED**2
    kernel = 1 / (2 * C2 - veff)

    T = numpy.zeros((mol.nao, mol.nao))
    eps_scal_ao = numpy.zeros((mol.nao, mol.nao))

    # Decent memory estimate
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, max_memory*.9-mem_now)

    for ao, weights, kern in _block_loop(mol, grid, kernel, deriv=1, max_memory=max_memory):
        T += numpy.einsum('xip,xiq,i->pq', ao[1:], ao[1:], weights*kern, optimize=True) * C2
        eps_scal_ao += numpy.einsum('xip,xiq,i->pq', ao[1:], ao[1:], weights*(kern**2), optimize=True) * C2

    zora_hcore = T + mol.intor('int1e_nuc')

    if not do_spin_orbit:
        return zora_hcore, eps_scal_ao, None

    ### Spin-orbit coupling section ###
    lib.logger.note(mf, 'Computing Spin-orbit coupling Hamiltonian')

    Hx = numpy.zeros((mol.nao,mol.nao))
    Hy = numpy.zeros_like(Hx)
    Hz = numpy.zeros_like(Hx)

    kernel *= veff/2.
    for ao, weights, kern in _block_loop(mol, grid, kernel, deriv=1, max_memory=max_memory):
        tx = numpy.einsum('ip,iq,i->pq', ao[2], ao[3], weights*kern, optimize=True)
        ty = numpy.einsum('ip,iq,i->pq', ao[3], ao[1], weights*kern, optimize=True)
        tz = numpy.einsum('ip,iq,i->pq', ao[1], ao[2], weights*kern, optimize=True)
        Hx += tx - tx.T
        Hy += ty - ty.T
        Hz += tz - tz.T

    spin_orbit_hamiltonian = numpy.block(
        [[Hz        ,  Hx - 1j*Hy],
         [Hx + 1j*Hy, -Hz        ]]
    )

    return zora_hcore, eps_scal_ao, spin_orbit_hamiltonian

scf.rhf.RHF.zora = add_zora_hcore_attribute_
scf.uhf.UHF.zora = add_zora_hcore_attribute_
scf.ghf.GHF.zora = add_zora_hcore_attribute_

dft.rks.RKS.zora = add_zora_hcore_attribute_
dft.uks.UKS.zora = add_zora_hcore_attribute_
dft.gks.GKS.zora = add_zora_hcore_attribute_

