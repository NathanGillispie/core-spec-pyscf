'''
Zeroth-Order Regular Approximation (ZORA) can be applied to any HF/KS
object by appending the zora method:

>>> mf = scf.RHF(mol).zora()
>>> mf.run()
>>> mol_eq = mf.Gradients().optimizer().kernel()
'''

from pyscf.scf import hf as _hf
from pyscf.zora.zora import zora, ZORAHelper, ZORA_SCF, compute_SO_coupling_matrix

_hf.SCF.zora = zora
