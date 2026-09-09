'''Zeroth-Order Regular Approximation (ZORA) for HF/KS objects.

Model-potential (MP) ZORA replaces the one-electron Hamiltonian with

    h = T^ZORA + V_nuc

where T^ZORA uses a tabulated atomic model potential rather than the
molecular KS potential. Typical usage:

>>> from pyscf import gto, scf
>>> import pyscf.zora
>>> mol = gto.M(...)
>>> mf = scf.RHF(mol).zora()
>>> mf.kernel()
>>> mol_eq = mf.Gradients().optimizer().kernel()

Spin-orbit MP-ZORA is available on GHF/GKS:

>>> mf = scf.GHF(mol).zora(spin_orbit=True)
'''

import numpy
import scipy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, ghf

from . import integrals
from .integrals import DEFAULT_GRID_LEVEL

__author__ = 'Nathan Gillispie'


def zora(mf, spin_orbit=False, grid_level=None, grid=None):
    '''Enable model-potential ZORA on an SCF object.

    Args:
        mf : :class:`pyscf.scf.hf.SCF`
            Mean-field object to wrap.

    Kwargs:
        spin_orbit : bool
            Include the MP-ZORA spin-orbit operator. Requires GHF or GKS.
        grid_level : int
            ZORA quadrature level (Treutler, no pruning). Default 8.
        grid : :class:`pyscf.dft.gen_grid.Grids` or None
            Optional pre-built grid. If given, ``grid_level`` is ignored.

    Returns:
        The same mean-field object with a ZORA mixin. Assign the result:
        ``mf = mf.zora()``.
    '''
    assert isinstance(mf, hf.SCF)

    if grid_level is None:
        grid_level = DEFAULT_GRID_LEVEL

    if spin_orbit and not isinstance(mf, ghf.GHF):
        logger.warn(mf, 'spin_orbit requires GHF/GKS; ignored.')
        spin_orbit = False

    if isinstance(mf, ZORA_SCF):
        mf.with_zora.spin_orbit = spin_orbit
        mf.with_zora.grid_level = grid_level
        if grid is not None:
            mf.with_zora.grids = grid
        mf.with_zora.reset()
        return mf

    obj = ZORA_SCF(mf, spin_orbit=spin_orbit, grid_level=grid_level, grid=grid)
    return lib.set_class(obj, (ZORA_SCF, mf.__class__))


class ZORAHelper(lib.StreamObject):
    '''Holds MP-ZORA options and cached one-electron integrals.'''

    _keys = {'mol', 'spin_orbit', 'grid_level', 'grids', 'max_memory'}

    def __init__(self,
                 mol,
                 spin_orbit=False,
                 grid_level=DEFAULT_GRID_LEVEL,
                 grid=None,
                 max_memory=None):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.spin_orbit = spin_orbit
        self.grid_level = grid_level
        self.grids = grid
        self.max_memory = max_memory if max_memory is not None else getattr(
            mol, 'max_memory', 4000)
        self._hcore = None
        self._hso = None
        self._coords = None
        self._eps_scal_ao = None
        self._dHso = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('grid_level = %s', self.grid_level)
        log.info('spin_orbit = %s', self.spin_orbit)
        return self

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._hcore = None
        self._hso = None
        self._coords = None
        self._eps_scal_ao = None
        self._dHso = None
        return self

    def _cache_valid(self, mol):
        if self._hcore is None or self._coords is None:
            return False
        if mol.nao != self._hcore.shape[-1]:
            return False
        return numpy.allclose(self._coords, mol.atom_coords())

    def _get_grid(self, mol):
        if self.grids is not None:
            return self.grids
        return integrals.build_zora_grid(mol, level=self.grid_level)

    def _build(self, mol):
        grid = self._get_grid(mol)
        veff = integrals.eval_model_potential(mol, grid.coords)
        kernel = integrals.zora_kernel(veff)
        max_memory = integrals.max_memory_mb(self)
        T, eps_scal_ao = integrals.eval_zora_T_and_eps(mol,
                                                       grid,
                                                       kernel,
                                                       max_memory=max_memory)
        hcore = T + mol.intor('int1e_nuc')
        self._hcore = hcore
        self._eps_scal_ao = eps_scal_ao
        self._coords = mol.atom_coords().copy()
        self._dHso = None

        if not self.spin_orbit:
            self._hso = None
            return

        log = logger.new_logger(self)
        log.note('Computing MP-ZORA spin-orbit Hamiltonian')
        Hx, Hy, Hz = integrals.eval_zora_SO(mol,
                                            grid,
                                            kernel * veff / 2.,
                                            max_memory=max_memory)
        self._hso = integrals.assemble_Hso(Hx, Hy, Hz)

    def get_hcore(self, mol=None):
        '''Scalar MP-ZORA hcore, shape (nao, nao).'''
        if mol is None:
            mol = self.mol
        if not self._cache_valid(mol):
            self._build(mol)
        return self._hcore

    def get_hso(self, mol=None):
        '''Spin-orbit Hamiltonian H_SO (multiplied by i in GHF hcore).'''
        if mol is None:
            mol = self.mol
        if not self.spin_orbit:
            return None
        if not self._cache_valid(mol) or self._hso is None:
            self._build(mol)
        return self._hso

    def hcore_deriv_generator(self, mol=None, deriv=1):
        from pyscf.zora import grad as zora_grad
        if deriv == 1:
            return zora_grad.hcore_grad_generator(self, mol)
        raise NotImplementedError('ZORA Hessian is not implemented')


class ZORA_SCF:
    '''Mixin that replaces ``get_hcore`` with MP-ZORA.'''

    __name_mixin__ = 'ZORA'

    _keys = {'with_zora'}

    def __init__(self,
                 mf,
                 spin_orbit=False,
                 grid_level=DEFAULT_GRID_LEVEL,
                 grid=None):
        self.__dict__.update(mf.__dict__)
        self.with_zora = ZORAHelper(mf.mol,
                                    spin_orbit=spin_orbit,
                                    grid_level=grid_level,
                                    grid=grid,
                                    max_memory=mf.max_memory)

    def undo_zora(self):
        '''Remove the ZORA mixin.'''
        obj = lib.view(self, lib.drop_class(self.__class__, ZORA_SCF))
        del obj.with_zora
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_zora.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_zora.reset(mol)
        return super().reset(mol)

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        self.with_zora.max_memory = self.max_memory
        hcore = self.with_zora.get_hcore(mol)
        if isinstance(self, ghf.GHF):
            hcore = scipy.linalg.block_diag(hcore, hcore)
            if self.with_zora.spin_orbit:
                hso = self.with_zora.get_hso(mol)
                hcore = hcore.astype(complex) + hso * 1j
                assert numpy.allclose(hcore, hcore.conj().T)
        return hcore

    def nuc_grad_method(self):
        from pyscf.zora.grad import make_grad_object
        return make_grad_object(self)

    Gradients = nuc_grad_method

    def _transfer_attrs_(self, dst):
        if self.with_zora and not hasattr(dst, 'with_zora'):
            logger.warn(
                self, 'Destination object of to_hf/to_ks is not a '
                'ZORA object. Convert dst to ZORA.')
            dst = dst.zora(spin_orbit=self.with_zora.spin_orbit,
                           grid_level=self.with_zora.grid_level)
        return hf.SCF._transfer_attrs_(self, dst)


def compute_SO_coupling_matrix(mf, X_s, w_s, X_t, w_t, frozen=None):
    '''Singlet-triplet SO coupling in the TD excitation basis.

    Arguments X_s, w_s are singlet excitation vectors and energies. X_t, w_t
    are the same but for triplet excitations.

    Only restricted references are supported.
    '''
    H_so = mf.with_zora.get_hso()
    assert H_so is not None
    H_so = H_so * 1j
    nroots = X_s.shape[0]
    nao = mf.mol.nao
    nocc, nvirt = X_s.shape[1:]

    occ_idx = numpy.where(mf.mo_occ == 2)[0]
    if hasattr(frozen, '__len__'):
        occ_idx = numpy.setxor1d(occ_idx, frozen)
    virt_idx = numpy.where(mf.mo_occ == 0)[0]
    active_idx = numpy.union1d(occ_idx, virt_idx)

    assert nocc == len(occ_idx)
    assert nvirt == len(virt_idx)

    Cmo = mf.mo_coeff[:, active_idx]

    h_z = Cmo.T @ H_so[:nao, :nao] @ Cmo
    h_m = Cmo.T @ H_so[:nao, nao:] @ Cmo
    h_p = Cmo.T @ H_so[nao:, :nao] @ Cmo

    oo = (slice(nocc), slice(nocc))
    vv = (slice(nocc, None), slice(nocc, None))
    s0 = slice(nroots)
    s1 = slice(nroots, 2 * nroots)
    s2 = slice(2 * nroots, 3 * nroots)
    s3 = slice(3 * nroots, None)

    h_oo = numpy.asarray([h_m[oo], h_z[oo], h_p[oo]])
    h_vv = numpy.asarray([h_m[vv], h_z[vv], h_p[vv]])

    H_ST = numpy.einsum('Iia,xab,Jib->xIJ', X_s, h_vv, X_t, optimize=True)
    H_ST -= numpy.einsum('Iia,xji,Jja->xIJ', X_s, h_oo, X_t, optimize=True)

    H_TS = numpy.einsum('Iia,xab,Jib->xIJ', X_t, h_vv, X_s, optimize=True)
    H_TS -= numpy.einsum('Iia,xji,Jja->xIJ', X_t, h_oo, X_s, optimize=True)

    H_TT = numpy.einsum('Iia,xab,Jib->xIJ', X_t, h_vv, X_t, optimize=True)
    H_TT += numpy.einsum('Iia,xji,Jja->xIJ', X_t, h_oo, X_t, optimize=True)

    H_so = numpy.diag(numpy.reshape((w_s, w_t, w_t, w_t), -1)).astype(complex)

    H_so[s1, s1] += H_TT[1]
    H_so[s3, s3] -= H_TT[1]

    H_so[s0, s1] += H_ST[2] / 2**.5
    H_so[s1, s0] += H_TS[0] / 2**.5

    H_so[s2, s3] -= H_TT[0] / 2**.5
    H_so[s3, s2] -= H_TT[2] / 2**.5

    H_so[s0, s2] += H_ST[1]
    H_so[s2, s0] += H_TS[1]

    H_so[s0, s3] -= H_ST[0] / 2**.5
    H_so[s1, s2] -= H_TT[0] / 2**.5
    H_so[s2, s1] -= H_TT[2] / 2**.5
    H_so[s3, s0] -= H_ST[2] / 2**.5

    w_so, X_so = numpy.linalg.eigh(H_so)
    w_so = w_so[:nroots]
    X_so = X_so[:nroots, :nroots].T
    return X_so, w_so
