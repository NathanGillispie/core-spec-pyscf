'''
Core-valence separation for TDDFT calculations.

Specify the occupied orbitals to excite from with ``core_idx``. This is
converted to PySCF's ``frozen`` attribute where valence occupied orbitals are
frozen.

>>> tdobj.core_idx = [0, 1, 2]
>>> tdobj.kernel()

Alternatively,

>>> tdobj.kernel(core_idx=[0, 1, 2])

Or,

>>> tdobj.core_valence(core_idx=[0, 1, 2])
>>> tdobj.kernel()

For UHF/UKS objects, specify a ``(alpha, beta)`` pair:

>>> tdobj.core_idx = ([0, 1], [0, 1])

You can also set ``tdobj.frozen`` directly using PySCF's convention.
'''

from pyscf.cvs import rhf
from pyscf.cvs import uhf
from pyscf.cvs import ghf
