'''Analytical nuclear gradients for model-potential ZORA.

Nuclear derivative of T_μν splits into:

(1) Pulay / AO-centre term (K held fixed), the ZORA analogue of int1e_ipkin.
    Nonzero only for AOs on atom A; PySCF symmetrizes (bra + ket).

(2) Kernel / model-potential term (AOs held fixed):

        ∂K/∂R_{A,α} = K^2 ∂V_MP/∂R_{A,α}

    which couples all AO pairs.

Nuclear attraction uses the NR treatment (int1e_ipnuc + int1e_iprinv).
Grid-weight response is neglected, as in default PySCF DFT gradients.

Spin-orbit: GHF contracts the spin-free ZORA hcore with the charge density;
Tr(D ∂(i H_SO)/∂R) is injected through extra_force.
'''

import numpy
from pyscf import gto, lib
from pyscf.grad import rhf as rhf_grad

from . import integrals


def nuc_grad_hcore(mol, grid=None, kernel=None, max_memory=None):
    '''Pulay part of the nuclear gradient of MP-ZORA hcore.

    Returns -⟨∂μ/∂r | T^ZORA + V_nuc | ν⟩ with K held fixed.
    '''
    if grid is None:
        grid = integrals.build_zora_grid(mol)
    if kernel is None:
        kernel = integrals.zora_kernel(
            integrals.eval_model_potential(mol, grid.coords))
    if max_memory is None:
        max_memory = integrals.max_memory_mb(mol)

    _, ipkin = integrals.eval_zora_T(mol,
                                     grid,
                                     kernel,
                                     deriv_bra=True,
                                     max_memory=max_memory)
    if mol._pseudo:
        raise NotImplementedError('Nuclear gradients for GTH PP')
    ipkin = ipkin + mol.intor('int1e_ipnuc', comp=3)
    if mol.has_ecp():
        ipkin = ipkin + mol.intor('ECPscalar_ipnuc', comp=3)
    return -ipkin


def hcore_grad_generator(zoraobj, mol=None):
    '''∂h_μν/∂R_A including Pulay T^ZORA, V_nuc, and the MP kernel term.

    Returns a function of atom index that yields the spin-free spatial
    derivative (3, nao, nao). When ``zoraobj.spin_orbit`` is set, the 2c
    ∂H_SO/∂R is stored on ``zoraobj._dHso`` for extra_force.
    '''
    if mol is None:
        mol = zoraobj.mol
    with_ecp = mol.has_ecp()
    ecp_atoms = set(mol._ecpbas[:, gto.ATOM_OF]) if with_ecp else ()
    aoslices = mol.aoslice_by_atom()

    if zoraobj.grids is not None:
        grid = zoraobj.grids
    else:
        grid = integrals.build_zora_grid(mol, level=zoraobj.grid_level)
    veff = integrals.eval_model_potential(mol, grid.coords)
    kernel = integrals.zora_kernel(veff)
    dveff = integrals.eval_dveff_all(mol, grid.coords)
    max_memory = integrals.max_memory_mb(zoraobj)
    h1 = nuc_grad_hcore(mol, grid=grid, kernel=kernel, max_memory=max_memory)
    dT = integrals.eval_zora_T_kernel_deriv(mol,
                                            grid,
                                            kernel,
                                            max_memory=max_memory,
                                            dveff=dveff)

    if zoraobj.spin_orbit:
        zoraobj._dHso = integrals.eval_zora_SO_grad(mol,
                                                    grid,
                                                    kernel,
                                                    veff,
                                                    max_memory=max_memory,
                                                    dveff=dveff)
    else:
        zoraobj._dHso = None

    def hcore_deriv(atm_id):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3)
            vrinv *= -mol.atom_charge(atm_id)
            if with_ecp and atm_id in ecp_atoms:
                vrinv += mol.intor('ECPscalar_iprinv', comp=3)
        vrinv[:, p0:p1] += h1[:, p0:p1]
        return vrinv + vrinv.transpose(0, 2, 1) + dT[atm_id]

    return hcore_deriv


def so_extra_force(mf_grad, atom_id, envs):
    '''Tr(D ∂(i H_SO)/∂R_A) over the full 2c GHF density.'''
    dHso = getattr(getattr(mf_grad.base, 'with_zora', None), '_dHso', None)
    if dHso is None:
        return 0
    dm0 = envs.get('dm0')
    if dm0 is None:
        return 0
    return numpy.einsum('xij,ji->x', 1j * dHso[atom_id], dm0).real


def make_grad_object(mf):
    '''Wrap the vacuum Gradients object with ZORA hcore / SO contributions.'''
    if isinstance(mf, rhf_grad.GradientsBase):
        mf = mf.base

    from pyscf.zora.zora import ZORA_SCF
    assert isinstance(mf, ZORA_SCF)

    vac_grad = mf.undo_zora().nuc_grad_method()
    vac_grad.base = mf
    return lib.set_class(ZORA_Gradients(vac_grad),
                         (ZORA_Gradients, vac_grad.__class__))


class ZORA_Gradients:
    '''Mixin that routes hcore derivatives through ``mf.with_zora``.'''

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)

    def hcore_generator(self, mol=None):
        return self.base.with_zora.hcore_deriv_generator(mol)

    def extra_force(self, atom_id, envs):
        return super().extra_force(atom_id, envs) + so_extra_force(
            self, atom_id, envs)
