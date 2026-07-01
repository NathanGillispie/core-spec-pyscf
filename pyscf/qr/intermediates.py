'''Immutable Casida-equation intermediates for quadratic response.

The left-hand side of the Casida-like equation depends only on the mean-field
object, not the excited-state pair. It is shared across every ``get_2tdm(i,
j)`` call and built lazily exactly once.

Each reference computes its own intermediates and assembles them via
:meth:`from_ab`. They are never written to checkpoint files.
'''

from dataclasses import dataclass

import numpy


@dataclass(frozen=True)
class CasidaIntermediates:
    '''Immutable, state-independent LHS of the QR Casida equation.

    Attributes
    ----------
    C : ndarray
        Generalized B matrix, shape ``(nocc, nvirt, nmo, nmo)``.
    Lambda : ndarray
        MO Hessian ``[[A, B], [B, A]]``.
    Delta : ndarray
        Generalized Metric ``[[I, 0], [0, -I]]``.
    '''

    C: numpy.ndarray
    Lambda: numpy.ndarray
    Delta: numpy.ndarray

    @property
    def nocc(self):
        return self.C.shape[0]

    @property
    def nvirt(self):
        return self.C.shape[1]

    @classmethod
    def from_ab(cls, C, A, B):
        '''Assemble the Casida LHS from reference-specific ``C``/``A``/``B``.

        Parameters
        ----------
        C : ndarray
            Generalized B matrix, shape ``(nocc, nvirt, nmo, nmo)``.
        A, B : ndarray
            Casida ``A`` and ``B`` matrices, shape ``(nocc*nvirt, nocc*nvirt)``.
        '''
        nocc, nvirt = C.shape[:2]
        Lambda = numpy.block([[A, B], [B, A]])
        Delta = numpy.eye(2 * nocc * nvirt)
        Delta[nocc * nvirt:] *= -1
        return cls(C, Lambda, Delta)
