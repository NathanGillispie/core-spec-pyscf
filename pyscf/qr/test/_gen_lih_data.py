'''One-off generator for the LiH reference intermediates used by test_LiH.py.

Run manually after any change to the reference math::

    python pyscf/qr/test/_gen_lih_data.py

This is the *only* remaining consumer of the ``reference/on-the-fly``
directory.  It writes ``lih_ref_intermediates.npz`` next to this file so that
the test suite no longer imports the reference at run time.  Once the reference
directory is removed for good, delete this script as well.

The stored MO basis (``mo_coeff``/``mo_energy``/``mo_occ``) is pinned by the
tests so that the amplitudes and intermediates are directly comparable without
SCF/Davidson phase or degeneracy ambiguity.
'''

import importlib.util
import os

import numpy
import pyscf
import pyscf.dft
import pyscf.tdscf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF_DIR = os.path.normpath(os.path.join(
    _HERE, '..', '..', '..', 'reference', 'on-the-fly'))
_OUT = os.path.join(_HERE, 'lih_ref_intermediates.npz')

_PAIRS = [(0, 3), (0, 1), (1, 3), (2, 3)]
_STATES = sorted({s for pair in _PAIRS for s in pair})


def _load_reference():
    import sys
    if _REF_DIR not in sys.path:
        sys.path.insert(0, _REF_DIR)
    spec = importlib.util.spec_from_file_location(
        '_qr_reference_otf', os.path.join(_REF_DIR, 'test.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ref_mod = _load_reference()

    mol = pyscf.gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='def2-svp', verbose=0)
    mf = pyscf.dft.RKS(mol, xc='PBE0').run()
    td = pyscf.tdscf.RPA(mf).set(nstates=4)
    td.kernel()

    payload = {
        'mo_coeff': mf.mo_coeff,
        'mo_energy': mf.mo_energy,
        'mo_occ': mf.mo_occ,
        'e': numpy.asarray(td.e),
    }
    for s in _STATES:
        x, y = td.xy[s]
        payload[f'x{s}'] = numpy.asarray(x)
        payload[f'y{s}'] = numpy.asarray(y)

    for (n, m) in _PAIRS:
        ref = ref_mod.reference_intermediates(mf, td, n, m)
        payload['C'] = ref['C']  # pair-independent
        payload[f'V_{n}_{m}'] = ref['V']
        payload[f'Knm_{n}_{m}'] = ref['Knm']
        payload[f'Pia_{n}_{m}'] = ref['Pia']
        payload[f'Qia_{n}_{m}'] = ref['Qia']
        payload[f'tdm_{n}_{m}'] = ref['tdm']

    numpy.savez_compressed(_OUT, **payload)
    print(f'wrote {_OUT} with {len(payload)} arrays')


if __name__ == '__main__':
    main()
