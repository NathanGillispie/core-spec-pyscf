import inspect

import pytest
from pyscf.tdscf.rhf import TDBase


requires_frozen = pytest.mark.skipif(
    'frozen' not in inspect.signature(TDBase.__init__).parameters,
    reason='This test requires PySCF TDSCF frozen-orbital support',
)
