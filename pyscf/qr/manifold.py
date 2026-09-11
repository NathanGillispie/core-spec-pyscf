'''Manifold objects for quadratic response.

These are essentially immutable Linear Response (LR) calculations with some metadata.
(De-)excitation vectors follow the PySCF TDSCF convention: ``xy`` is a tuple of
length ``nstates``, each element ``(x, y)`` where ``y`` is ``0`` or ``(0,0)``
for TDA and numpy.ndarray for RPA.

Each manifold carries ``mol`` and ``mo_coeff`` references shared with sibling
manifolds in a QR calculation.  The mf object lives on the :class:`QR` driver only.
Users don't have to deal with these directly. They are designed to be immutable.

Example
-------
>>> td = RPA(mf, frozen=core_idx).set(nstates=80)
>>> td.kernel()
>>> manifold = Manifold.from_tdobj(td)
>>> manifold.occ_idx   # active occupied MO indices in full MO basis
>>> len(manifold.xy)   # 80
'''

import json
from dataclasses import dataclass

import numpy


def _active_occ_idx(mf, frozen):
    '''Return active occupied MO indices for a linear-response calculation.

    ``frozen`` follows the TDSCF ``frozen`` convention (int count or index list).
    '''
    occ_all = numpy.where(mf.mo_occ > 0)[0]
    if frozen is None:
        return occ_all.copy()

    moidx = numpy.ones(mf.mo_occ.size, dtype=bool)
    if isinstance(frozen, (int, numpy.integer)):
        moidx[:frozen] = False
    elif hasattr(frozen, '__len__'):
        moidx[list(frozen)] = False
    else:
        raise NotImplementedError(f'frozen={frozen!r}')

    active = numpy.where(moidx)[0]
    return numpy.intersect1d(occ_all, active, assume_unique=True)


def _decode_xy(raw):
    '''Restore ``xy`` from a checkpoint payload.'''
    assert isinstance(raw, list) and isinstance(raw[0], (list, tuple))
    assert len(raw[0]) == 2
    assert isinstance(raw[0][1], (list, tuple, int))

    def _deserialize_y(z):
        # RHF/GHF TDA
        if z == 0:
            return 0
        if len(z) == 2:
            # UHF TDA
            if z[0] == 0 and z[1] == 0:
                return (0, 0)
        # RPA
        return numpy.asarray(z)

    return tuple((numpy.asarray(x), _deserialize_y(y)) for x, y in raw)


def _encode_xy(xy):
    '''Serialize ``xy`` for JSON checkpoint storage.'''

    def _serialize_y(z):
        if type(z) == numpy.ndarray:
            return z.tolist()
        if type(z) == int:
            return 0
        if type(z) == tuple:
            return 0
        raise ValueError('Unknown type of y in manifold xy!')

    return [[x.tolist(), _serialize_y(y)] for x, y in xy]


@dataclass(frozen=True)
class Manifold:
    '''Immutable linear-response manifold for one active occupied subspace.

    Attributes
    ----------
    mol : Mole
        Molecular object.
    mo_coeff : ndarray
        MO coefficients in the AO basis.
    mo_occ : ndarray
        Mean-field occupation numbers in the MO basis (shared with ``mf``).
    occ_idx : ndarray
        1D int array of active occupied MO indices in the full MO basis.
    e : ndarray
        Excitation energies in Hartree, shape ``(nstates,)``.
    xy : tuple of (ndarray, ndarray or int or tuple)
        PySCF-style amplitudes per state.
    '''

    mol: object
    mo_coeff: numpy.ndarray
    mo_occ: numpy.ndarray
    occ_idx: numpy.ndarray
    e: numpy.ndarray
    xy: tuple

    def __post_init__(self):
        object.__setattr__(self, 'mo_coeff', numpy.asarray(self.mo_coeff))
        object.__setattr__(self, 'mo_occ', numpy.asarray(self.mo_occ))
        object.__setattr__(self, 'occ_idx',
                           numpy.asarray(self.occ_idx, dtype=int))
        object.__setattr__(self, 'e', numpy.asarray(self.e))
        object.__setattr__(self, 'xy', self.xy)

        if len(self.xy) != len(self.e):
            raise ValueError(
                f'len(xy)={len(self.xy)} does not match len(e)={len(self.e)}')

        if len(self.xy) == 0:
            raise ValueError(
                'No excited states found in tdobj. Did you freeze all the orbitals?')

        has_y = {isinstance(y, numpy.ndarray) for x, y in self.xy}
        if len(has_y) != 1:
            raise ValueError('All states in a manifold must be TDA or RPA.')

    @classmethod
    def from_tdobj(cls, tdobj):
        '''Build a manifold from a converged TDSCF object.

        Parameters
        ----------
        tdobj : TDSCF
            A TDA, RPA, or TDHF object.  If ``tdobj.kernel()`` has not run,
            it is called here.

        Returns
        -------
        Manifold

        Notes
        -----
        ``occ_idx`` is derived from ``tdobj.frozen`` and ``tdobj._scf.mo_occ``.
        '''
        if tdobj.e is None or tdobj.xy is None:
            tdobj.kernel()

        mf = tdobj._scf
        frozen = getattr(tdobj, 'frozen', None)
        occ_idx = _active_occ_idx(mf, frozen)

        return cls(
            mol=mf.mol,
            mo_coeff=mf.mo_coeff,
            mo_occ=numpy.asarray(mf.mo_occ),
            occ_idx=occ_idx,
            e=numpy.asarray(tdobj.e),
            xy=tdobj.xy,
        )

    def get_aligned_xy(self, state):
        '''Pad compact TD amplitudes onto the full MO (occ, virt) basis.

        Rows index all occupied MOs (``mo_occ > 0``); columns index all virtual
        MOs (``mo_occ == 0``).  Compact amplitudes are placed at rows
        ``occ_idx``; columns follow the standard PySCF ordering of virtual MOs.

        Parameters
        ----------
        state : int
            0-based excited-state index.

        Returns
        -------
        x, y : ndarray
            Padded amplitudes with shape ``(nocc, nvirt)`` for the full MO
            basis.  Rows and columns outside the active LR subspace are zero.
        '''
        if state < 0 or state >= len(self.e):
            raise IndexError(
                f'state={state} out of range for {len(self.e)} states')
        x, y = self.xy[state]
        mo_occ = self.mo_occ
        occ_full = numpy.where(mo_occ > 0)[0]
        nvirt_full = int(numpy.count_nonzero(mo_occ == 0))
        x_pad = numpy.zeros((len(occ_full), nvirt_full), dtype=x.dtype)
        y_pad = numpy.zeros((len(occ_full), nvirt_full), dtype=x.dtype)
        row_pos = numpy.searchsorted(occ_full, self.occ_idx)
        ncol = x.shape[1]
        if ncol > nvirt_full:
            raise ValueError(
                f'compact x has {ncol} virtual columns but mo_occ has '
                f'only {nvirt_full} virtual orbitals')
        x_pad[row_pos, :ncol] = x
        if isinstance(y, numpy.ndarray):
            y_pad[row_pos, :ncol] = y
        return x_pad, y_pad

    def __call__(self, state):
        '''Return excitation energy and aligned amplitudes for one state.

        Parameters
        ----------
        state : int
            0-based excited-state index.

        Returns
        -------
        e : float
            Excitation energy in Hartree.
        xy : tuple of ndarray
            ``(x, y)`` padded onto the full MO basis (see :meth:`get_aligned_xy`).
        '''
        if state < 0 or state >= len(self.e):
            raise IndexError(
                f'state={state} out of range for {len(self.e)} states')
        return float(self.e[state]), self.get_aligned_xy(state)

    def dump(self):
        '''Serialize linear-response data to a JSON string.

        ``mol`` and ``mo_coeff`` are not included; restore them via
        :meth:`loads` with a live ``mf`` object.

        Returns
        -------
        str
            JSON string suitable for writing to a QR checkpoint file.
        '''
        payload = {
            'occ_idx': self.occ_idx.tolist(),
            'e': self.e.tolist(),
            'xy': _encode_xy(self.xy),
        }
        return json.dumps(payload)

    @classmethod
    def loads(cls, s, mf):
        '''Reconstruct a :class:`Manifold` from :meth:`dump` output.

        Parameters
        ----------
        s : str
            JSON string from :meth:`dump`.
        mf : SCF
            Mean-field object providing ``mol`` and ``mo_coeff``.
        '''
        payload = json.loads(s)
        return cls(
            mol=mf.mol,
            mo_coeff=mf.mo_coeff,
            mo_occ=numpy.asarray(mf.mo_occ),
            occ_idx=payload['occ_idx'],
            e=payload['e'],
            xy=_decode_xy(payload['xy']),
        )

    def dumps(self):
        '''Alias for :meth:`dump`.'''
        return self.dump()


def gxc_tensor_shape(manifold_n, manifold_m, nvirt):
    '''Shape of the 6-index :math:`g_\\text{xc}` tensor for a QR calculation.

    Returns
    -------
    tuple of int
        ``(nocc, nvirt, nocc_n, nvirt_n, nocc_m, nvirt_m)``
    '''
    nocc = int(numpy.count_nonzero(manifold_n.mo_occ > 0))
    nocc_n = len(manifold_n.occ_idx)
    nocc_m = len(manifold_m.occ_idx)
    nvirt_n = manifold_n.xy[0][0].shape[1]
    nvirt_m = manifold_m.xy[0][0].shape[1]
    return (nocc, nvirt, nocc_n, nvirt_n, nocc_m, nvirt_m)


def check_shared_reference(tdobj_n, tdobj_m):
    '''Verify that two TDSCF objects share the same ``mol`` and ``mo_coeff``.

    Raises
    ------
    ValueError
        If references differ.
    '''
    mf_n = tdobj_n._scf
    mf_m = tdobj_m._scf
    if mf_n.mol is not mf_m.mol:
        raise ValueError('TDSCF objects must share the same mol object')
    if mf_n.mo_coeff is not mf_m.mo_coeff:
        raise ValueError('TDSCF objects must share the same mo_coeff object')
