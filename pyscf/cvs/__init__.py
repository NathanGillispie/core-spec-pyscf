'''
Core-valence separation for TDDFT calculations
Use by specifying the core orbitals with the core_idx attribute

>>> tdobj.core_idx = [0,1,2]
>>> tdobj.kernel()
'''

from pyscf.cvs import rhf
from pyscf.cvs import uhf
from pyscf.cvs import ghf

