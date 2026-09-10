'''Focused edge-case tests for ZORA coverage.'''

from types import SimpleNamespace
from importlib import import_module

import numpy
import pytest
import pyscf
import pyscf.zora

from pyscf.zora import compute_SO_coupling_matrix
from pyscf.zora import grad as zora_grad
from pyscf.zora import integrals
from pyscf.zora.test.common import GRAD_GRID_LEVEL, make_hf_mol

zora_module = import_module('pyscf.zora.zora')


def test_nuc_grad_hcore_defaults_and_pseudo_rejection(monkeypatch):
    class FakeMol:
        _pseudo = False

        def intor(self, name, comp=None):
            assert name == 'int1e_ipnuc'
            assert comp == 3
            return numpy.zeros((3, 1, 1))

        def has_ecp(self):
            return False

    mol = FakeMol()
    grid = SimpleNamespace(coords=numpy.zeros((1, 3)))
    monkeypatch.setattr(integrals, 'build_zora_grid', lambda obj: grid)
    monkeypatch.setattr(integrals, 'eval_model_potential',
                        lambda obj, coords: numpy.zeros(coords.shape[0]))
    monkeypatch.setattr(integrals, 'zora_kernel',
                        lambda veff: numpy.ones_like(veff))
    monkeypatch.setattr(integrals, 'max_memory_mb', lambda obj: 2000)
    monkeypatch.setattr(
        integrals,
        'eval_zora_T',
        lambda obj, grid, kernel, deriv_bra, max_memory:
        (numpy.zeros((1, 1)), numpy.zeros((3, 1, 1))),
    )

    assert zora_grad.nuc_grad_hcore(mol).shape == (3, 1, 1)
    mol._pseudo = True
    with pytest.raises(NotImplementedError, match='GTH PP'):
        zora_grad.nuc_grad_hcore(mol)


def test_hcore_grad_generator_uses_default_mol_and_custom_grid(monkeypatch):
    mol = make_hf_mol()
    grid = SimpleNamespace(coords=numpy.zeros((1, 3)))
    nao = mol.nao
    monkeypatch.setattr(integrals, 'eval_model_potential',
                        lambda obj, coords: numpy.zeros(coords.shape[0]))
    monkeypatch.setattr(integrals, 'zora_kernel',
                        lambda veff: numpy.ones_like(veff))
    monkeypatch.setattr(integrals, 'eval_dveff_all',
                        lambda obj, coords: numpy.zeros((obj.natm, 3, 1)))
    monkeypatch.setattr(integrals, 'max_memory_mb', lambda obj: 2000)
    monkeypatch.setattr(
        zora_grad,
        'nuc_grad_hcore',
        lambda obj, **kwargs: numpy.zeros((3, nao, nao)),
    )
    monkeypatch.setattr(
        integrals,
        'eval_zora_T_kernel_deriv',
        lambda obj, grid, kernel, max_memory, dveff:
        numpy.zeros((obj.natm, 3, nao, nao)),
    )

    zoraobj = SimpleNamespace(
        mol=mol,
        grids=grid,
        spin_orbit=False,
        max_memory=2000,
    )
    deriv = zora_grad.hcore_grad_generator(zoraobj)
    assert deriv(0).shape == (3, nao, nao)


def test_so_extra_force_without_density():
    dHso = numpy.zeros((1, 3, 1, 1))
    grad_method = SimpleNamespace(
        base=SimpleNamespace(with_zora=SimpleNamespace(_dHso=dHso)),
    )
    assert zora_grad.so_extra_force(grad_method, 0, {}) == 0


def test_zora_rewrap_with_grid():
    mol = make_hf_mol()
    mf = pyscf.scf.RHF(mol).zora(grid_level=GRAD_GRID_LEVEL)
    grid = object()
    assert mf.zora(grid=grid) is mf
    assert mf.with_zora.grids is grid


def test_zora_helper_cache_edge_cases(monkeypatch):
    mol = make_hf_mol()
    helper = zora_module.ZORAHelper(mol)
    helper._hcore = numpy.zeros((1, 1))
    helper._coords = mol.atom_coords().copy()
    assert helper._cache_valid(mol) is False

    hcore = numpy.eye(mol.nao)
    helper._hcore = hcore
    helper._coords = mol.atom_coords().copy()
    assert helper.get_hcore() is hcore

    helper = zora_module.ZORAHelper(mol, spin_orbit=True)
    helper._hcore = numpy.eye(mol.nao)
    helper._coords = mol.atom_coords().copy()
    hso = numpy.ones((2 * mol.nao, 2 * mol.nao), dtype=complex)
    monkeypatch.setattr(helper, '_build', lambda obj: setattr(helper, '_hso', hso))
    assert helper.get_hso() is hso


def test_so_coupling_frozen_orbital_branch():
    mol = SimpleNamespace(nao=3)
    mf = SimpleNamespace(
        mol=mol,
        mo_occ=numpy.array([2., 2., 0.]),
        mo_coeff=numpy.eye(3),
        with_zora=SimpleNamespace(
            get_hso=lambda: numpy.zeros((6, 6), dtype=complex),
        ),
    )
    x_s = numpy.ones((1, 1, 1))
    x_t = numpy.ones((1, 1, 1))
    x_so, w_so = compute_SO_coupling_matrix(
        mf, x_s, numpy.array([1.]), x_t, numpy.array([2.]), frozen=[0],
    )
    assert x_so.shape == (1, 1)
    assert w_so.shape == (1,)
