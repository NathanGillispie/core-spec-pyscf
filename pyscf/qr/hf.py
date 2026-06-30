'''Base quadratic response driver.'''

from pyscf import lib, scf

from pyscf.qr import chkfile as qr_chkfile
from pyscf.qr.manifold import (
    Manifold,
    check_shared_reference,
)


def qr_class_for_mf(mf):
    '''Return the QR driver class appropriate for a mean-field object.'''
    from pyscf.qr.ghf import GQR
    from pyscf.qr.rhf import RQR
    from pyscf.qr.uhf import UQR

    if isinstance(mf, scf.ghf.GHF):
        return GQR
    if isinstance(mf, scf.uhf.UHF):
        return UQR
    if isinstance(mf, scf.hf.RHF):
        return RQR
    raise TypeError(
        f'Unsupported mean-field type {type(mf).__name__} for QR')


class QR(lib.StreamObject):
    '''Base quadratic response driver.

    Construct from converged (or pending) TDSCF objects.  Linear response is
    run during initialization if needed; the tdobjs are not retained.

    Parameters
    ----------
    tdobj_n : TDSCF
        Primary linear-response object.
    tdobj_m : TDSCF, optional
        Secondary linear-response object.  Defaults to ``tdobj_n``.
    chkfile : str, optional
        Checkpoint path.  When omitted, inherits from ``tdobj_n._scf.chkfile``.
    precompute_gxc : bool, optional
        When True, :meth:`kernel` stores the 6-index ``Gxc`` tensor in
        memory for reuse by :meth:`get_2tdm``.

    Notes
    -----
    **Checkpoint workflow:** call :meth:`save` after initialization (LR
    complete) and before :meth:`kernel` (QR stage).  Resume with
    :meth:`from_chk` and a live ``mf`` object.

    Reference-specific subclasses (e.g. :class:`RQR`) implement ``g_xc``
    contraction and 2TDM evaluation.

    When calculating excited-to-excited state properties, excitations go from
    manifold N to M.
    '''

    _keys = {
        'verbose', 'stdout', 'max_memory', 'mol', 'chkfile',
        'precompute_gxc', 'response_type', 'manifold_n', 'manifold_m',
    }

    precompute_gxc = False

    def __init__(self, tdobj_n, tdobj_m=None, *, chkfile=None,
                 precompute_gxc=False):
        if tdobj_m is None:
            tdobj_m = tdobj_n
        elif tdobj_m is not tdobj_n:
            check_shared_reference(tdobj_n, tdobj_m)

        mf = tdobj_n._scf
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self._scf = mf
        self.mol = mf.mol
        self.max_memory = mf.max_memory
        self.chkfile = chkfile if chkfile is not None else mf.chkfile
        self.precompute_gxc = bool(precompute_gxc)

        self._manifold_n = Manifold.from_tdobj(tdobj_n)
        if tdobj_m is tdobj_n:
            self._manifold_m = self._manifold_n
        else:
            self._manifold_m = Manifold.from_tdobj(tdobj_m)

        self._set_response_type()
        self._init_gxc()

    @classmethod
    def _from_restored(cls, chkfile, mf, *, precompute_gxc=False):
        '''Build a QR driver from a checkpoint file and a live mean-field object.

        Manifold data are read from *chkfile*; reference-specific behavior is
        determined by the concrete subclass (selected via :func:`qr_class_for_mf`
        in :meth:`from_chk`).
        '''
        manifold_n = qr_chkfile.load_manifold_n(chkfile, mf)
        manifold_m = qr_chkfile.load_manifold_m(chkfile, mf, manifold_n)

        self = cls.__new__(cls)
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self._scf = mf
        self.mol = mf.mol
        self.max_memory = mf.max_memory
        self.chkfile = chkfile
        self.precompute_gxc = bool(precompute_gxc)

        self._manifold_n = manifold_n
        self._manifold_m = manifold_m

        self._set_response_type()
        self._init_gxc()
        return self

    @classmethod
    def from_chk(cls, chkfile, mf, *, precompute_gxc=False):
        '''Restore a :class:`QR` instance from a checkpoint file.

        This can only be used following :meth:``save``.

        Parameters
        ----------
        chkfile : str
            HDF5 checkpoint path.
        mf : SCF
            Mean-field object for the calculation.  The appropriate
            reference-specific driver (e.g. :class:`RQR`) is chosen from
            the type of *mf*.
        precompute_gxc : bool, optional
            Forwarded to the restored driver.

        Returns
        -------
        QR subclass object.
        '''
        driver_cls = qr_class_for_mf(mf)
        return driver_cls._from_restored(
            chkfile, mf, precompute_gxc=precompute_gxc)

    def save(self, chkfile=None):
        '''Write checkpoint to disk.

        Call after initialization (linear response complete) and before
        :meth:`kernel`.  Stores manifold LR results only; pass ``mf`` to
        :meth:`from_chk` to restore reference data.  ``Gxc`` is not saved.

        The intended purpose is to pause and inspect LR results before moving on
        to QR. This may be necessary to set energy bounds on plots, etc.

        Parameters
        ----------
        chkfile : str, optional
            Destination path.  Defaults to ``self.chkfile``.

        Returns
        -------
        self
        '''
        qr_chkfile.save_qr(self, chkfile=chkfile)
        return self

    def _set_response_type(self):
        self.response_type = self._infer_response_type(self._manifold_n)

        if self._manifold_m is not self._manifold_n:
            m_type = self._infer_response_type(self._manifold_m)
            if m_type != self.response_type:
                raise ValueError(
                    'manifold_n and manifold_m must use the same response '
                    f'type (got {self.response_type!r} and {m_type!r})')

    @staticmethod
    def _infer_response_type(manifold):
        _, y0 = manifold.xy[0]
        return 'tda' if y0 is None else 'rpa'

    def _init_gxc(self):
        '''Hook for reference-specific Gxc backend setup.'''
        raise NotImplementedError('_init_gxc called from base QR class. '
                                  'Subclass override needed.')

    @property
    def manifold_n(self):
        return self._manifold_n

    @property
    def manifold_m(self):
        return self._manifold_m

    @property
    def mf(self):
        return self._scf

    @property
    def mo_coeff(self):
        return self._scf.mo_coeff

    @property
    def mo_occ(self):
        return self._scf.mo_occ

    @property
    def mo_energy(self):
        return self._scf.mo_energy

    def _sanity_check_2tdm(self, i, j):
        '''Validate state indices and return excitation energies.'''
        if i < 0 or i >= len(self._manifold_n.e):
            raise IndexError(f'i={i} out of range for manifold_n '
                             f'({len(self._manifold_n.e)} states)')
        if j < 0 or j >= len(self._manifold_m.e):
            raise IndexError(f'j={j} out of range for manifold_m '
                             f'({len(self._manifold_m.e)} states)')

        e_i = float(self._manifold_n.e[i])
        e_j = float(self._manifold_m.e[j])
        lib.logger.new_logger(self).info(
            'QR 2TDM: manifold_n[%d] -> manifold_m[%d], omega=%.6f Ha',
            i, j, e_j - e_i)
