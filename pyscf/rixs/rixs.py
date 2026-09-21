'''RIXS calculation driver using the Kramers-Heisenberg equation.

Stores QR and mean-field references and provides RIXS-specific response
quantities built from them.
'''

import numpy

from pyscf import lib
from pyscf.qr.dipole import compute_dipole_mo as _compute_dipole_mo
from pyscf.qr.rhf import RQR

from pyscf.rixs import chkfile as _chkfile
from pyscf.rixs.response import (
    _normalize_state_indices,
    ground_transition_dipoles as _ground_transition_dipoles,
)

class RIXS(lib.StreamObject):
    '''Container for a mean-field reference and a QR calculation.'''

    _keys = {
        'mf',
        'qr',
        'mol',
        'chkfile',
        'dipole_mo',
    }

    def __init__(self, mf, qr, *, chkfile=None):
        if not isinstance(qr, RQR):
            raise NotImplementedError(
                f'Only RQR driver supported. Got {type(qr).__name__}.')
        if qr.mf is not mf:
            raise ValueError('mf and qr must use the same mean-field object')

        self.mf = mf
        self.qr = qr
        self.mol = mf.mol
        self.chkfile = chkfile if chkfile is not None else qr.chkfile
        self.dipole_mo = None

    def _get_dipole_mo(self):
        '''Return MO dipole integrals, computing them on first use.'''
        if self.dipole_mo is None:
            self.dipole_mo = _compute_dipole_mo(
                self.mol,
                self.qr.mo_coeff,
            )
        return self.dipole_mo

    def save(self, chkfile=None):
        '''Write this RIXS calculation to a checkpoint file.'''
        _chkfile.save_rixs(self, chkfile=chkfile)
        return self

    def ground_transition_dipoles(self, states=None):
        '''Return ground-to-intermediate transition dipoles.

        Parameters
        ----------
        states : array_like of int, optional
            0-based indices into ``qr.manifold_n``.  Defaults to all
            intermediate states.

        Returns
        -------
        ndarray
            Transition dipoles with shape ``(3, len(states))``.
        '''
        return _ground_transition_dipoles(
            self.mol,
            self.qr.mo_coeff,
            self.qr.mo_occ,
            self.qr.manifold_n,
            states,
            dipole_mo=self._get_dipole_mo(),
        )

    def transition_dipoles(self, intermediate_states=None, final_states=None):
        '''Return selected intermediate-to-final transition dipoles.

        Parameters
        ----------
        intermediate_states : array_like of int, optional
            0-based indices into ``qr.manifold_n``.  Defaults to all states.
        final_states : array_like of int, optional
            0-based indices into ``qr.manifold_m``.  Defaults to all states.

        Returns
        -------
        ndarray
            Transition dipoles with shape
            ``(3, len(final_states), len(intermediate_states))``.

        Notes
        -----
        The QR stage is evaluated only after the requested states have been
        validated.  Eager QR objects are prepared here as well; repeated
        preparation is a no-op.
        '''
        n_states = _normalize_state_indices(
            intermediate_states,
            len(self.qr.manifold_n.e),
            'intermediate_states',
        )
        f_states = _normalize_state_indices(
            final_states,
            len(self.qr.manifold_m.e),
            'final_states',
        )

        dtype = numpy.result_type(self.qr.mo_coeff.dtype, numpy.float64)
        if len(n_states) == 0 or len(f_states) == 0:
            return numpy.empty(
                (3, len(f_states), len(n_states)),
                dtype=dtype,
            )

        self.qr.kernel()
        dipole_mo = self._get_dipole_mo()
        log = lib.logger.new_logger(self)

        dipoles = numpy.empty(
            (3, len(f_states), len(n_states)),
            dtype=dtype,
        )
        for f_pos, f_state in enumerate(f_states):
            for n_pos, n_state in enumerate(n_states):
                log.info(
                    'RIXS dipole f=%d, n=%d',
                    f_state,
                    n_state,
                )
                tdm = self.qr.get_2tdm(int(n_state), int(f_state))
                dipoles[:, f_pos, n_pos] = self.qr.transition_dipole(
                    tdm,
                    dipole_mo=dipole_mo,
                )

        return dipoles

    @classmethod
    def from_chk(cls, chkfile, mf, *, precompute_gxc=False):
        '''Restore a RIXS calculation into a live MF template.'''
        return _chkfile.load_rixs(
            chkfile,
            mf,
            precompute_gxc=precompute_gxc,
        )
