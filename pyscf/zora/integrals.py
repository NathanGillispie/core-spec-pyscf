'''Model-potential ZORA integrals.

The scalar operator is density-independent:

    T^ZORA = p · [c^2 / (2 c^2 - V_MP)] p

so the AO matrix (quadrature on a DFT grid) is

    T_μν = c^2 ∫ K(r) ∇χ_μ(r) · ∇χ_ν(r) dr,
    K(r) = 1 / (2 c^2 - V_MP(r)).

V_MP is a superposition of tabulated atomic model potentials
(van Wüllen / ADF-style), not the molecular KS potential.
'''

import numpy
import scipy
from pyscf import dft, lib
from pyscf.data.nist import LIGHT_SPEED

from .modbas2c import modbas

_SQRTPI = numpy.sqrt(numpy.pi)
_RMIN = 1e-16
# Hessian (xx,xy,xz / xy,yy,yz / xz,yz,zz) of (∂χ/∂x, ∂χ/∂y, ∂χ/∂z)
_HESS_NABLA = ((4, 5, 6), (5, 7, 8), (6, 8, 9))
# (ao_i, ao_j) for t_yz, t_zx, t_xy used to build Hx, Hy, Hz
_SO_PAIRS = ((2, 3), (3, 1), (1, 2))

DEFAULT_GRID_LEVEL = 8


def block_loop(mol, grids, kernel, deriv=0, max_memory=2000):
    '''Loop over grids by blocks, yielding (ao, weight, kernel_block).'''
    ngrids = grids.coords.shape[0]
    nao = mol.nao
    comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
    assert kernel.shape[0] == ngrids

    from pyscf.dft.gen_grid import BLKSIZE
    blksize = int(max_memory * 1e6 / ((comp + 1) * nao * 8 * BLKSIZE))
    blksize = max(4, min(blksize, ngrids // BLKSIZE + 1, 1200)) * BLKSIZE
    assert blksize % BLKSIZE == 0

    buf = dft.numint._empty_aligned(comp * blksize * nao)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[ip0:ip1]
        weight = grids.weights[ip0:ip1]
        kern = kernel[ip0:ip1]
        ao = dft.numint.eval_ao(mol,
                                coords,
                                deriv=deriv,
                                cutoff=grids.cutoff,
                                out=buf)
        yield ao, weight, kern


def modbas_for_mol(mol):
    '''Atomic MP parameters (c, a) for each atom, indexed by nuclear charge.'''
    return [numpy.asarray(modbas[z]) for z in mol.atom_charges()]


def build_zora_grid(mol, level=None):
    '''Treutler grid, no pruning (reduces quadrature noise near nuclei).'''
    grid = dft.gen_grid.Grids(mol)
    grid.prune = None
    grid.level = DEFAULT_GRID_LEVEL if level is None else level
    grid.build(with_non0tab=False)
    return grid


def eval_model_potential(mol, coords, c_a=None):
    r'''V_MP(r) on a grid.

    V_MP(r) = Σ_A [ Σ_i c_{A,i} erf(a_{A,i} R_A)/R_A  -  Z_A / R_A ]
    '''
    if c_a is None:
        c_a = modbas_for_mol(mol)
    veff = numpy.zeros(coords.shape[0])
    charges = mol.atom_charges()
    for atom_xyz, (c, a), Z in zip(mol.atom_coords(), c_a, charges):
        RPA = numpy.sqrt(numpy.sum((coords - atom_xyz)**2, axis=1))
        RPA = numpy.maximum(RPA, _RMIN)
        outer = numpy.outer(a, RPA)
        erf_over = scipy.special.erf(outer) / outer
        veff += numpy.einsum('i,i,ip->p', c, a, erf_over, optimize=True)
        veff -= Z / RPA
    return veff


def eval_dveff_dRA(coords, atom_xyz, c, a, Z):
    r'''∂V_MP(r)/∂R_{A,α} for one atom, shape (3, ngrid).

    ∂V_A/∂R_{A,α} = (r_α - R_{A,α}) / R_A^3
        * { Σ_i c_i [erf(a_i R) - (2 a_i R / √π) exp(-a_i^2 R^2)] - Z }
    '''
    PA = coords - atom_xyz
    R = numpy.sqrt(numpy.sum(PA**2, axis=1))
    R = numpy.maximum(R, _RMIN)
    a = numpy.asarray(a)
    c = numpy.asarray(c)
    aR = numpy.outer(a, R)
    erf_aR = scipy.special.erf(aR)
    gauss = numpy.exp(-aR**2)
    two_aR_sqrtpi = (2.0 / _SQRTPI) * aR * gauss
    bracket = numpy.einsum('i,ip->p', c, erf_aR - two_aR_sqrtpi) - Z
    return PA.T * (bracket / R**3)


def eval_dveff_all(mol, coords, c_a=None):
    '''∂V_MP/∂R_A on a grid, shape (natm, 3, ngrid).'''
    if c_a is None:
        c_a = modbas_for_mol(mol)
    dveff = numpy.empty((mol.natm, 3, coords.shape[0]))
    charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    for ia in range(mol.natm):
        c, a = c_a[ia]
        dveff[ia] = eval_dveff_dRA(coords, atom_coords[ia], c, a, charges[ia])
    return dveff


def zora_kernel(veff):
    return 1.0 / (2.0 * LIGHT_SPEED**2 - veff)


def max_memory_mb(obj):
    mem_now = lib.current_memory()[0]
    return max(2000, getattr(obj, 'max_memory', 4000) * .9 - mem_now)


def eval_zora_T(mol, grid, kernel, deriv_bra=False, max_memory=2000):
    r'''Integrate T^ZORA and optionally the Pulay bra derivative.

    T_μν = c^2 ∫ K ∇χ_μ · ∇χ_ν
    ipkin_αμν = c^2 ∫ K ∇(∂χ_μ/∂r_α) · ∇χ_ν   (int1e_ipkin analogue)

    Returns T, and ipkin if deriv_bra else None.
    '''
    C2 = LIGHT_SPEED**2
    nao = mol.nao
    T = numpy.zeros((nao, nao))
    ipkin = numpy.zeros((3, nao, nao)) if deriv_bra else None
    ao_deriv = 2 if deriv_bra else 1
    for ao, weights, kern in block_loop(mol,
                                        grid,
                                        kernel,
                                        deriv=ao_deriv,
                                        max_memory=max_memory):
        wK = weights * kern
        T += numpy.einsum('xip,xiq,i->pq', ao[1:4], ao[1:4], wK,
                          optimize=True) * C2
        if deriv_bra:
            ipkin += numpy.einsum(
                'xip,iq,i->xpq', ao[[4, 5, 6]], ao[1], wK, optimize=True) * C2
            ipkin += numpy.einsum(
                'xip,iq,i->xpq', ao[[5, 7, 8]], ao[2], wK, optimize=True) * C2
            ipkin += numpy.einsum(
                'xip,iq,i->xpq', ao[[6, 8, 9]], ao[3], wK, optimize=True) * C2
    return T, ipkin


def eval_zora_T_and_eps(mol, grid, kernel, max_memory=2000):
    '''T^ZORA and the unused picture-change-like eps_scal_ao matrix.'''
    C2 = LIGHT_SPEED**2
    nao = mol.nao
    T = numpy.zeros((nao, nao))
    eps_scal_ao = numpy.zeros((nao, nao))
    for ao, weights, kern in block_loop(mol,
                                        grid,
                                        kernel,
                                        deriv=1,
                                        max_memory=max_memory):
        wK = weights * kern
        T += numpy.einsum('xip,xiq,i->pq', ao[1:4], ao[1:4], wK,
                          optimize=True) * C2
        eps_scal_ao += numpy.einsum('xip,xiq,i->pq',
                                    ao[1:4],
                                    ao[1:4],
                                    weights * (kern**2),
                                    optimize=True) * C2
    return T, eps_scal_ao


def eval_zora_T_kernel_deriv(mol, grid, kernel, max_memory=2000, dveff=None):
    r'''Kernel (model-potential) contribution to ∂T_μν/∂R_A.

    (∂T_μν/∂R_{A,α})|_AO = c^2 ∫ K^2 (∂V_MP/∂R_{A,α}) ∇χ_μ · ∇χ_ν dr

    Returns array of shape (natm, 3, nao, nao).
    '''
    C2 = LIGHT_SPEED**2
    if dveff is None:
        dveff = eval_dveff_all(mol, grid.coords)

    dT = numpy.zeros((mol.natm, 3, mol.nao, mol.nao))
    ip0 = 0
    for ao, weights, kern in block_loop(mol,
                                        grid,
                                        kernel,
                                        deriv=1,
                                        max_memory=max_memory):
        ip1 = ip0 + weights.shape[0]
        fac = dveff[:, :, ip0:ip1] * (weights * kern**2)
        dT += numpy.einsum(
            'xip,xiq,ayi->aypq', ao[1:4], ao[1:4], fac, optimize=True) * C2
        ip0 = ip1
    dT = 0.5 * (dT + dT.transpose(0, 1, 3, 2))
    return dT


def assemble_Hso(Hx, Hy, Hz):
    '''2c H_SO from Cartesian pieces. Leading axes of Hx, Hy, Hz are preserved.'''
    nao = Hx.shape[-1]
    lead = Hx.shape[:-2]
    out = numpy.empty(lead + (2 * nao, 2 * nao), dtype=complex)
    out[..., :nao, :nao] = Hz
    out[..., :nao, nao:] = Hx - 1j * Hy
    out[..., nao:, :nao] = Hx + 1j * Hy
    out[..., nao:, nao:] = -Hz
    return out


def eval_zora_SO(mol, grid, kappa, max_memory=2000):
    '''Real antisymmetric Hx, Hy, Hz matching the energy-level SO integrals.'''
    nao = mol.nao
    t = numpy.zeros((3, nao, nao))
    for ao, weights, kern in block_loop(mol,
                                        grid,
                                        kappa,
                                        deriv=1,
                                        max_memory=max_memory):
        wK = weights * kern
        for k, (i, j) in enumerate(_SO_PAIRS):
            t[k] += numpy.einsum('ip,iq,i->pq',
                                 ao[i],
                                 ao[j],
                                 wK,
                                 optimize=True)
    Hx = t[0] - t[0].T
    Hy = t[1] - t[1].T
    Hz = t[2] - t[2].T
    return Hx, Hy, Hz


def eval_zora_SO_grad(mol, grid, kernel, veff, max_memory=2000, dveff=None):
    r'''∂H_SO/∂R_A, shape (natm, 3, 2nao, 2nao).

    Matches the matrix that is multiplied by i in GHF hcore. Pulay uses AO
    Hessians; the kernel term uses

        ∂κ/∂R_{A,α} = c^2 K^2 ∂V_MP/∂R_{A,α}.
    '''
    C2 = LIGHT_SPEED**2
    nao = mol.nao
    natm = mol.natm
    if dveff is None:
        dveff = eval_dveff_all(mol, grid.coords)
    kappa = kernel * veff / 2.
    aoslices = mol.aoslice_by_atom()

    ip_bra = numpy.zeros((3, 3, nao, nao))
    ip_ket = numpy.zeros((3, 3, nao, nao))
    dt = numpy.zeros((3, natm, 3, nao, nao))
    ip0 = 0
    for ao, weights, kap in block_loop(mol,
                                       grid,
                                       kappa,
                                       deriv=2,
                                       max_memory=max_memory):
        ip1 = ip0 + weights.shape[0]
        wK = weights * kap
        dkap = dveff[:, :, ip0:ip1] * (weights * kernel[ip0:ip1]**2 * C2)
        for k, (i, j) in enumerate(_SO_PAIRS):
            hess_i = ao[list(_HESS_NABLA[i - 1])]
            hess_j = ao[list(_HESS_NABLA[j - 1])]
            ip_bra[k] += numpy.einsum('xip,iq,i->xpq',
                                      hess_i,
                                      ao[j],
                                      wK,
                                      optimize=True)
            ip_ket[k] += numpy.einsum('ip,xiq,i->xpq',
                                      ao[i],
                                      hess_j,
                                      wK,
                                      optimize=True)
            dt[k] += numpy.einsum('ip,iq,ayi->aypq',
                                  ao[i],
                                  ao[j],
                                  dkap,
                                  optimize=True)
        ip0 = ip1

    dHcart = numpy.empty((3, natm, 3, nao, nao))
    for k in range(3):
        ipH = ip_bra[k] - ip_ket[k].transpose(0, 2, 1)
        pulay = numpy.zeros((natm, 3, nao, nao))
        for ia in range(natm):
            p0, p1 = aoslices[ia, 2:]
            tmp = numpy.zeros((3, nao, nao))
            tmp[:, p0:p1] -= ipH[:, p0:p1]
            pulay[ia] = tmp - tmp.transpose(0, 2, 1)
        kern = dt[k] - dt[k].transpose(0, 1, 3, 2)
        dHk = pulay + kern
        dHcart[k] = 0.5 * (dHk - dHk.transpose(0, 1, 3, 2))

    dHso = assemble_Hso(dHcart[0], dHcart[1], dHcart[2])
    dHso = 0.5 * (dHso - numpy.moveaxis(dHso, -1, -2).conj())
    return dHso
