'''Checkpoint I/O for :class:`RIXS`.'''

from pyscf import lib
from pyscf.scf import chkfile as scf_chkfile

from pyscf.qr import chkfile as qr_chkfile
from pyscf.qr.hf import QR as QRBase


_ZORA_KEY = 'rixs/zora'


def _zora_settings(mf):
    '''Return the serializable ZORA settings on *mf*, if enabled.'''
    helper = getattr(mf, 'with_zora', None)
    if helper is None:
        return {'enabled': False}

    return {
        'enabled': True,
        'spin_orbit': bool(helper.spin_orbit),
        'grid_level': int(helper.grid_level),
    }


def _restore_zora(chkfile, mf):
    '''Apply the checkpoint's ZORA setting to a mean-field template.'''
    settings = lib.chkfile.load(chkfile, _ZORA_KEY)
    if settings is None:
        return mf

    from pyscf.zora.zora import ZORA_SCF

    if settings.get('enabled', False):
        if isinstance(mf, ZORA_SCF):
            mf = mf.undo_zora()
        import pyscf.zora  # noqa: F401  # attaches .zora to SCF objects
        mf = mf.zora(spin_orbit=bool(settings['spin_orbit']),
                     grid_level=int(settings['grid_level']))
    elif isinstance(mf, ZORA_SCF):
        mf = mf.undo_zora()
    return mf


def save_rixs(rixsobj, chkfile=None):
    '''Write the MF, QR, dipole, and ZORA state to *chkfile*.'''
    chkfile = chkfile or rixsobj.chkfile
    if not chkfile:
        return

    mf = rixsobj.mf
    zora_settings = _zora_settings(mf)
    mf_to_save = mf.undo_zora() if zora_settings['enabled'] else mf

    scf_chkfile.dump_scf(mf_to_save.mol,
                         chkfile,
                         mf_to_save.e_tot,
                         mf_to_save.mo_energy,
                         mf_to_save.mo_coeff,
                         mf_to_save.mo_occ)
    lib.chkfile.save(chkfile, _ZORA_KEY, zora_settings)

    qr_chkfile.save_qr(rixsobj.qr, chkfile=chkfile)
    rixsobj.dipole_mo = qr_chkfile.save_dipole_mo(chkfile, mf)


def load_rixs(chkfile, mf, *, precompute_gxc=False):
    '''Restore a :class:`RIXS` object into a live MF template.'''
    from pyscf.rixs.rixs import RIXS

    mf = _restore_zora(chkfile, mf)
    mf.update_from_chk(chkfile)
    mf.chkfile = chkfile

    qr = QRBase.from_chk(chkfile, mf, precompute_gxc=precompute_gxc)
    obj = RIXS(mf, qr, chkfile=chkfile)
    obj.dipole_mo = qr_chkfile.load_dipole_mo(chkfile)
    return obj
