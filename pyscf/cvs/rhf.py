import pyscf
from pyscf.tdscf.rhf import TDHF, TDA
import numpy

from pyscf.lib import logger
from pyscf import ao2mo

def core_valence(self, core_idx=None):
    '''This can be manually called to perform the CVS.
    Don't try this with something silly like fractional occupation numbers.'''
    if hasattr(self, 'core_idx'):
        core_idx = self.core_idx
    if core_idx is None:
        raise RuntimeError('Core orbitals not specified')

    self.check_sanity() # scf object exists and ran
    scf = self._scf

    if type(core_idx) is int:
        core_idx = [core_idx]

    core_idx = numpy.asarray(core_idx)
    scf.mol.nelec = (len(core_idx), len(core_idx))

    occ_idx = numpy.where(scf.mo_occ!=0)
    if not all(numpy.isin(core_idx, occ_idx)):
        print('Listed core orbitals aren\'t even occupied!')
    delete_idx = numpy.setxor1d(occ_idx, core_idx)

    scf.mo_occ = numpy.delete(scf.mo_occ, delete_idx, 0)
    scf.mo_coeff = numpy.delete(scf.mo_coeff, delete_idx, axis=1)
    scf.mo_energy = numpy.delete(scf.mo_energy, delete_idx, 0)

def get_ab_no_fxc(mf, mo_energy=None, mo_coeff=None, mo_occ=None, add_hf=True):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ai||jb)
    B[i,a,j,b] = (ai||bj)

    Ref: Chem Phys Lett, 256, 454
    '''
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    # assert (mo_coeff.dtype == numpy.double)

    assert mo_coeff.dtype == numpy.float64
    mol = mf.mol
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = numpy.hstack((orbo,orbv))

    e_ia = mo_energy[viridx] - mo_energy[occidx,None]
    a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = numpy.zeros_like(a)

    if add_hf:
        eri_mo = ao2mo.general(mol, [orbo,mo,mo,mo], compact=False)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        a += numpy.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc]) * 2
        a -= numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:])

        b += numpy.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * 2
        b -= numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:])

    return a, b

def direct_diag_tda_kernel(self, x0=None, nstates=None):
    '''TDA diagonalization solver'''
    cpu0 = (logger.process_clock(), logger.perf_counter())
    self.check_sanity()
    self.dump_flags()
    if nstates is None:
        nstates = self.nstates
    else:
        self.nstates = nstates
    mol = self.mol

    log = logger.Logger(self.stdout, self.verbose)

    a, _ = self.get_ab()
    nocc, nvirt, _, _ = a.shape
    a = a.reshape(nocc*nvirt, nocc*nvirt)

    self.e, self.xy = numpy.linalg.eigh(a)
    self.converged = [True]

    if self.chkfile:
        pyscf.lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
        pyscf.lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

    log.timer('TDA', *cpu0)
    self._finalize()
    return self.e, self.xy

@pyscf.lib.with_doc(TDHF.kernel.__doc__)
def rpa_kernel(self, **kwargs):
    '''Monkey-patched TDHF/TDDFT kernel for CVS'''
    if 'core_idx' in kwargs.keys():
        self.core_idx = kwargs.pop('core_idx')
    if hasattr(self, 'core_idx'):
        self.core_valence()

    if 'no_fxc' in kwargs.keys():
        self.no_fxc = kwargs.pop('no_fxc')
    if hasattr(self, 'no_fxc'):
        if self.no_fxc:
            raise NotImplementedError('no_fxc not in RPA kernels yet')

    if 'direct_diag' in kwargs.keys():
        self.direct_diag = kwargs.pop('direct_diag')
    if hasattr(self, 'direct_diag'):
        if self.direct_diag:
            raise NotImplementedError('direct_diag not in RPA kernels yet')
    return self._old_kernel(**kwargs)

@pyscf.lib.with_doc(TDA.kernel.__doc__)
def tda_kernel(self, **kwargs):
    '''Monkey-patched TDA kernel for CVS'''
    if 'core_idx' in kwargs.keys():
        self.core_idx = kwargs.pop('core_idx')
    if hasattr(self, 'core_idx'):
        self.core_valence()

    if 'no_fxc' in kwargs.keys():
        self.no_fxc = kwargs.pop('no_fxc')
    if hasattr(self, 'no_fxc'):
        if self.no_fxc:
            self.get_ab = get_ab_no_fxc

    if 'direct_diag' in kwargs.keys():
        self.direct_diag = kwargs.pop('direct_diag')
    if hasattr(self, 'direct_diag'):
        if self.direct_diag:
            return direct_diag_tda_kernel(self, **kwargs)
    return self._old_kernel(**kwargs)

TDHF.core_valence = core_valence
TDHF._old_kernel = TDHF.kernel
TDHF.kernel = rpa_kernel

TDA.core_valence = core_valence
TDA._old_kernel = TDA.kernel
TDA.kernel = tda_kernel


