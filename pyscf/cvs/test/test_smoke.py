def test_pyscf_import():
    import pyscf


def test_plugin_import():
    import pyscf.cvs


def test_plugin_hasattr():
    from pyscf.tdscf.rhf import TDBase, TDHF
    import pyscf.cvs
    assert hasattr(TDBase, 'cvs')
    assert hasattr(TDHF, 'cvs')
    assert not hasattr(TDHF, '_old_kernel')


def test_doc():
    from pyscf.tdscf.rhf import TDHF
    import pyscf.cvs
    assert hasattr(TDHF, '__doc__')
