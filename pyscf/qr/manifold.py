'''Manifold objects for quadratic response.

These are essentially immutable Linear Response (LR) calculations with some metadata.
(De-)excitation vectors follow the PySCF TDSCF convention: ``xy`` is a tuple of
length ``nstates``, each element ``(x, y)`` where ``y`` is ``None`` for TDA and
an ndarray for RPA.

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
>>> manifold.xy[0][1] is not None  # RPA has Y amplitudes
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


def _is_tda_y(y):
    '''Return True when *y* carries no RPA de-excitation amplitudes.'''
    if y is None:
        return True
    y = numpy.asarray(y)
    return y.ndim == 0


def _normalize_xy(pyscf_xy):
    '''Convert PySCF ``tdobj.xy`` to an immutable tuple of ``(x, y)`` pairs.

    TDA states store ``y=None`` (PySCF itself uses scalar ``0``).
    '''
    out = []
    for x, y in pyscf_xy:
        x = numpy.asarray(x)
        if _is_tda_y(y):
            out.append((x, None))
        else:
            out.append((x, numpy.asarray(y)))
    return tuple(out)


def _decode_xy(raw):
    '''Restore ``xy`` from a checkpoint payload (tuple or legacy ndarray).'''
    if isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple)):
        first = raw[0]
        if len(first) == 2 and (first[1] is None or isinstance(first[1], list)):
            return tuple(
                (numpy.asarray(x), None if y is None else numpy.asarray(y))
                for x, y in raw
            )

    arr = numpy.asarray(raw)
    if arr.ndim == 4 and arr.shape[1] == 2:
        return tuple((arr[i, 0], arr[i, 1]) for i in range(arr.shape[0]))
    if arr.ndim == 3:
        return tuple((arr[i], None) for i in range(arr.shape[0]))
    raise ValueError(f'cannot decode xy payload with shape {arr.shape}')


def _encode_xy(xy):
    '''Serialize ``xy`` for JSON checkpoint storage.'''
    return [
        [x.tolist(), None if y is None else y.tolist()]
        for x, y in xy
    ]


@dataclass(frozen=True)
class Manifold:
    '''Immutable linear-response manifold for one active occupied subspace.

    Attributes
    ----------
    mol : Mole
        Molecular object.
    mo_coeff : ndarray
        MO coefficients in the AO basis.
    occ_idx : ndarray
        1D int array of active occupied MO indices in the full MO basis.
    e : ndarray
        Excitation energies in Hartree, shape ``(nstates,)``.
    xy : tuple of (ndarray, ndarray or None)
        PySCF-style amplitudes per state.  ``y is None`` for TDA.
    nstates : int or None
        Number of roots requested from the LR calculation (metadata).
    '''

    mol: object
    mo_coeff: numpy.ndarray
    occ_idx: numpy.ndarray
    e: numpy.ndarray
    xy: tuple
    nstates: int | None = None

    def __post_init__(self):
        object.__setattr__(self, 'mo_coeff', numpy.asarray(self.mo_coeff))
        object.__setattr__(self, 'occ_idx', numpy.asarray(self.occ_idx, dtype=int))
        object.__setattr__(self, 'e', numpy.asarray(self.e))

        xy = tuple(
            (numpy.asarray(x), None if y is None else numpy.asarray(y))
            for x, y in self.xy
        )
        object.__setattr__(self, 'xy', xy)

        if len(xy) != len(self.e):
            raise ValueError(
                f'len(xy)={len(xy)} does not match len(e)={len(self.e)}')

        has_y = {y is not None for x, y in xy}
        if len(has_y) != 1:
            raise ValueError(
                'all states in a manifold must be TDA or RPA')

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
        nstates = getattr(tdobj, 'nstates', None)

        return cls(
            mol=mf.mol,
            mo_coeff=mf.mo_coeff,
            occ_idx=occ_idx,
            e=numpy.asarray(tdobj.e),
            xy=_normalize_xy(tdobj.xy),
            nstates=nstates,
        )

    def get_xy_state(self, state):
        '''Return amplitudes for one excited state as ``(x, y)``.

        For TDA, ``y`` is ``None``.

        Parameters
        ----------
        state : int
            0-based index into ``e`` and ``xy``.
        '''
        if state < 0 or state >= len(self.e):
            raise IndexError(f'state={state} out of range for {len(self.e)} states')
        return self.xy[state]

    def get_x_y(self, state):
        '''Return ``(x, y)`` ndarrays; TDA uses ``y = zeros_like(x)``.'''
        x, y = self.get_xy_state(state)
        if y is None:
            y = numpy.zeros_like(x)
        return x, y

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
            'nstates': self.nstates,
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
            occ_idx=payload['occ_idx'],
            e=payload['e'],
            xy=_decode_xy(payload['xy']),
            nstates=payload.get('nstates'),
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
    nocc = len(union_occ_idx(manifold_n, manifold_m))
    nocc_n = len(manifold_n.occ_idx)
    nocc_m = len(manifold_m.occ_idx)
    nvirt_n = manifold_n.xy[0][0].shape[1]
    nvirt_m = manifold_m.xy[0][0].shape[1]
    return (nocc, nvirt, nocc_n, nvirt_n, nocc_m, nvirt_m)


def union_occ_idx(*manifolds):
    '''Union of active occupied indices across one or more manifolds.

    Used when aligning amplitudes from different manifolds onto a common
    active-occ basis for Gxc contraction or 2TDM evaluation.
    '''
    if not manifolds:
        raise ValueError('At least one manifold is required')
    out = manifolds[0].occ_idx
    for man in manifolds[1:]:
        out = numpy.union1d(out, man.occ_idx)
    return out


def align_xy(manifold, state, union_occ):
    '''Pad one state's amplitudes onto a union active-occ basis.

    Storage in :class:`Manifold` remains compact; padding is applied only
    when cross-manifold QR quantities require a shared ``(nocc_union, nvirt)``
    indexing.

    Parameters
    ----------
    manifold : Manifold
    state : int
        0-based excited-state index.
    union_occ : ndarray
        Sorted union of ``occ_idx`` values from all manifolds involved.

    Returns
    -------
    x, y : ndarray
        Padded amplitudes with shape ``(len(union_occ), nvirt)``.  Rows of
        ``union_occ`` not in ``manifold.occ_idx`` are zero.
    '''
    x, y = manifold.get_x_y(state)
    nvirt = x.shape[1]
    union_occ = numpy.asarray(union_occ, dtype=int)
    x_pad = numpy.zeros((len(union_occ), nvirt), dtype=x.dtype)
    pos = numpy.searchsorted(union_occ, manifold.occ_idx)
    x_pad[pos, :] = x
    y_pad = numpy.zeros((len(union_occ), nvirt), dtype=y.dtype)
    _, y_raw = manifold.get_xy_state(state)
    if y_raw is not None:
        y_pad[pos, :] = y_raw
    return x_pad, y_pad


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


def check_shared_manifolds(*manifolds):
    '''Verify that manifolds share the same ``mol`` and ``mo_coeff`` objects.

    Raises
    ------
    ValueError
        If references differ.
    '''
    if len(manifolds) < 2:
        return
    mol = manifolds[0].mol
    mo_coeff = manifolds[0].mo_coeff
    for man in manifolds[1:]:
        if man.mol is not mol:
            raise ValueError('Manifolds must share the same mol object')
        if man.mo_coeff is not mo_coeff:
            raise ValueError('Manifolds must share the same mo_coeff object')
