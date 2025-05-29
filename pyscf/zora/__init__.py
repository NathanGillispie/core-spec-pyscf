'''
Zeroth-Order Regular Approximation (ZORA) can be applied to any HF/KS
object by appending the zora method:

>>> mf = scf.RHF(mol).zora()
>>> mf.run()
'''

from pyscf.zora import ZORA

