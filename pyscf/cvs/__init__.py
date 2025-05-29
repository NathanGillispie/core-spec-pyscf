'''
Core-valence separation for TDDFT calculations
Use by specifying the core orbitals with the core_idx attribute

>>> tdobj.core_idx = [0,1,2]
>>> tdobj.kernel()

Alternatively,

>>> tdobj.kernel(core_idx=[0,1,2])

Or,

>>> tdobj.core_valence(core_idx=[0,1,2])
>>> tdobj.kernel()

For UHF/UKS objects, specify tuples

>>> tdobj.core_idx = ([0,1], [0,1])
'''

from pyscf.cvs import rhf
from pyscf.cvs import uhf
from pyscf.cvs import ghf

