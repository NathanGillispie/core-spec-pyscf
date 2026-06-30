'''Quadratic response driver for restricted RHF/RKS references.'''

import numpy
from scipy.linalg import block_diag

from pyscf import lib, scf
from pyscf import ao2mo
from pyscf.tdscf.rhf import _charge_center

from pyscf.qr.hf import QR
from pyscf.qr.manifold import gxc_tensor_shape

def _precompute_gxc(mf, G, occ_idx_n, occ_idx_m):
    '''Fill the 6-index :math:`g_\\text{xc}` buffer in place.'''
    raise NotImplementedError('Gxc precomputation is not implemented yet.')


def _get_2tdm_diag_block(x1, x2, y1, y2):
    '''Return 2TDM where off-diagonal blocks are 0.'''
    goo = -x1@x2.T
    gvv = x2.T@x1
    if y1 is not None:
        goo -= y2@y1.T
        gvv += y1.T@y2
    return block_diag(goo, gvv)


def _get_pq(C, Knm, V, x1, x2, y1, y2):
    '''Create |P,Q>: RHS of casida-like eq. for QR'''
    nocc, nvirt = C.shape[:2]
    oo = (slice(nocc),slice(nocc))
    vv = (slice(nocc,None),slice(nocc,None))

    Pia = numpy.einsum('iapq,pq->ia', C, Knm) + V
    Qia = numpy.einsum('iapq,pq->ia', C, Knm.T) + V

    Ha = numpy.einsum('iaqp,ia->pq', C, x1)
    Hb = numpy.einsum('iapq,ia->pq', C, x2)
    if y1 is not None:
        Ha += numpy.einsum('iapq,ia->pq', C, y1)
        Hb += numpy.einsum('iaqp,ia->pq', C, y2)

    Pia += x2@Ha[vv].T - Ha[oo].T@x2
    Qia += x1@Hb[vv] - Hb[oo]@x1

    if y1 is None:
        return Pia, Qia

    Pia += y1@Hb[vv].T - Hb[oo].T@y1
    Qia += y2@Ha[vv] - Ha[oo]@y2
    return Pia, Qia


def _compute_c(mf):
    '''Returns a matrix of shape (nocc, nvirt, nmo, nmo).

    This is a generalization of the B matrix in linear response.

    The (o,v,o,v) and (o,v,v,o) blocks are used to compute A & B like in linear
    response. The (o,v,o,o) and (o,v,v,v) blocks are used for QR, but I
    don't separate them for simplicity. Nothing goes to waste here.
    '''
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mol = mf.mol

    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = numpy.hstack((orbo,orbv))

    C = numpy.zeros((nocc,nvir,nmo,nmo))

    def add_hf_(c1, hyb=1):
        eri_mo = ao2mo.general(mol, [mo,mo,mo,mo], compact=False)
        eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
        c1 += numpy.einsum('iapq->iapq',eri_mo[:nocc,nocc:]) * 2
        c1 -= numpy.einsum('iqpa->iapq',eri_mo[:nocc,:,:,nocc:]) * hyb

    if not isinstance(mf, scf.hf.KohnShamDFT):
        add_hf_(C)
        return C

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    add_hf_(C, hyb)
    if omega != 0:  # For RSH
        with mol.with_range_coulomb(omega):
            eri_mo = ao2mo.general(mol, [orbo,mo,mo,orbv], compact=False)
            eri_mo = eri_mo.reshape(nocc,nmo,nmo,nvir)
            k_fac = alpha - hyb
            C -= numpy.einsum('iqpa->iapq', eri_mo) * k_fac

    xctype = ni._xc_type(mf.xc)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]
    mem_now = lib.current_memory()[0]
    ## TODO: update memory usage
    max_memory = max(2000, mf.max_memory*.8-mem_now)

    if xctype=='GGA':
        ao_deriv = 1
        for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
            rho = make_rho(0, ao, mask, xctype)

            fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
            wfxc = fxc * weight

            rho = numpy.einsum('xrp,pi->xri', ao, mo)
            rho_xx = numpy.einsum('xrp,rq->xrpq', rho, rho[0])
            rho_xx[1:4] += numpy.einsum('rp,xrq->xrpq', rho[0], rho[1:4])
            rho_xx = numpy.transpose(rho_xx, (1,2,3,0))
            rho_ov = rho_xx[:,:nocc,nocc:]

            w_xx = numpy.einsum('rpqx,xyr->ypqr', rho_xx, wfxc, optimize=True)
            iapq = numpy.einsum('xpqr,riax->iapq', w_xx, rho_ov, optimize=True) * 2
            C += iapq
    elif xctype=='LDA':
        ao_deriv = 0
        for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
            rho = make_rho(0, ao, mask, xctype)
            fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
            wfxc = fxc[0,0] * weight

            rho = numpy.einsum('rp,pi->ri', ao, mo)
            rho_xx = numpy.einsum('rp,rq->rpq', rho, rho)
            rho_ov = rho_xx[:,:nocc,nocc:]
            w_ov = numpy.einsum('ria,r->ria', rho_ov, wfxc)
            iapq = numpy.einsum('rpq,ria->iapq', rho_xx, w_ov) * 2
            C += iapq
    else:
        raise NotImplementedError(f'xctype = {xctype}')

    return C


def _get_ab_from_c(C, mo_energy):
    '''Returns the regular A,B matrices given C from :meth:``_compute_c``'''
    nocc, nvirt = C.shape[:2]
    e_ia = mo_energy[nocc:] - mo_energy[:nocc,None]
    A = numpy.diag(e_ia.ravel()).reshape(nocc,nvirt,nocc,nvirt)
    B = numpy.zeros_like(A)
    A += C[:,:,nocc:,:nocc].transpose((0,1,3,2))
    B += C[:,:,:nocc,nocc:]
    return numpy.reshape((A,B), (2,nocc*nvirt,nocc*nvirt))


class LazyGxc:
    '''Compute :math:``g_\\text{xc}`` new each time for each state pair.

    Avoids storing the 6-index tensor but repeats grid work per call.

    Notes
    -----
    Generally faster than :class:``EagerGxc`` when you only need a few calls.
    '''

    def contract_v(self, mf, xpy1, xpy2, mo_occ=None):
        '''Return the ``V_ia`` vector given two excitated states.

        ``xpy1`` represents the sum of ``x1`` and ``y1``: the first excitation.
        '''
        if (xpy1.shape[0] < nocc) or (xpy2.shape[0] < nocc):
            raise NotImplementedError('Frozen orbitals not yet implemented for '
                                      'LazyGxc evaluation!') # TODO: this

        if mo_occ is None: mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff

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

        G = numpy.zeros((nocc,nvir))

        if not isinstance(mf, scf.hf.KohnShamDFT):
            return G

        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
        xctype = ni._xc_type(mf.xc)

        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

        if xctype=='GGA':
            ao_deriv = 1
            for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho = make_rho(0, ao, mask, xctype)
                gxc = ni.eval_xc_eff(mf.xc, rho, deriv=3, xctype=xctype)[3]

                rho_o = numpy.einsum('xrp,pi->xri', ao, orbo)
                rho_v = numpy.einsum('xrp,pi->xri', ao, orbv)
                rho_ov = numpy.einsum('xrp,rq->xrpq', rho_o, rho_v[0])
                rho_ov[1:4] += numpy.einsum('rp,xrq->xrpq', rho_o[0], rho_v[1:4])
                # This makes later contractions much faster
                rho_ov = numpy.transpose(rho_ov, (1,2,3,0))

                # Convert excitation in MO basis to grid domain.
                xp1 = numpy.einsum('riax,ia->rx',rho_ov,xpy1)
                xp2 = numpy.einsum('riax,ia->rx',rho_ov,xpy2)

                # '...r,r...->...' looks like matrix multiplication. This is fast!
                wgxc = gxc * weight
                w_ov = numpy.einsum('riax,xyzr->yziar', rho_ov, wgxc, optimize=True)
                iajb = numpy.einsum('rx,xyiar->yiar', xp1, w_ov, optimize=True)
                iajbkc = numpy.einsum('xiar,rx->ia', iajb, xp2, optimize=True) * 4
                G += iajbkc
        elif xctype=='LDA':
            ao_deriv = 0
            for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                lib.logger.warn(mf, 'LazyGxc contraction for LDA functionals: '
                                    'running untested code!') # TODO: test this

                rho = make_rho(0, ao, mask, xctype)
                gxc = ni.eval_xc_eff(mf.xc, rho, deriv=3, xctype=xctype)[3]

                rho = numpy.einsum('rp,pi->ri', ao, mo)
                rho_xx = numpy.einsum('rp,rq->rpq', rho, rho)
                rho_ov = rho_xx[:,:nocc,nocc:]

                xp1 = numpy.einsum('ria,ia->r', rho_ov, xpy1)
                xp2 = numpy.einsum('ria,ia->r', rho_ov, xpy2)

                wgxc = gxc[0,0,0] * weight
                w_ov = numpy.einsum('ria,r->iar', rho_ov, wgxc, optimize=True)
                iajb = numpy.einsum('r,iar->iar', xp1, w_ov, optimize=True)
                iajbkc = numpy.einsum('iar,r->ia', iajb, xp2, optimize=True) * 4
                G += iajbkc
        else:
            raise NotImplementedError(f'xctype = {xctype}')

        return G


class EagerGxc:
    '''Contract :math:`g_\\text{xc}` from a precomputed 6-index tensor in memory.

    ``G`` has shape ``(nocc, nvirt, nocc_n, nvirt_n, nocc_m, nvirt_m)`` and is
    filled in place by :meth:`RQR.kernel` when ``precompute_gxc=True``.  The
    tensor is never written to checkpoint files.

    The last two pairs of indices will differ when any of the two excited states
    used frozen orbitals for the linear response calculation. That assumption
    allows a lot of space/computation to be saved. It also means that no padding
    or reshaping is required from excitation vectors before :meth:``contract_v``.
    '''

    def __init__(self, G):
        self.G = G

    def contract_v(self, mf, xpy1, xpy2, mo_occ=None):
        return numpy.einsum('iajbkc,jb,kc->ia', self.G, xpy1, xpy2, optimize=True)


def transition_dipole(qrobj, tdm):
    '''Returns transition dipole moment given excited-to-excited state 2TDM.

    Parameters
    ----------
    self : QR object
    tdm : ndarray
        Excited-to-excited state transition density matrix.

    Returns
    -------
    tdip : ndarray
        Dipole (x,y,z) components.
    '''
    mol = qrobj.mol
    coeff = qrobj.mo_coeff
    with mol.with_common_orig(_charge_center(mol)):
        ints_ao = mol.intor_symmetric('int1e_r', comp=3)
    ints = numpy.einsum('xpq,pi,qa->xia', ints_ao, coeff, coeff)
    return numpy.einsum('xpq,pq->x', ints, tdm14)


def oscillator_strength(qrobj, i, j, tdm=None):
    '''Returns oscillator strength given excited-to-excited state 2TDM.

    Parameters
    ----------
    self : QR object
    i : int
        i-th state of manifold_n
    j : int
        j-th state of manifold_m
    tdm : ndarray (optional)
        Excited-to-excited state transition density matrix. If not provided,
        it will be computed.

    Returns
    -------
    osc : float
        Oscillator strength
    '''
    if tdm is None:
        tdm = qrobj.get_2tdm(i, j)
    tdip = qrobj.transition_dipole(tdm)
    ei, _ = qrobj._manifold_n(i)
    ej, _ = qrobj._manifold_m(j)
    return float(2./3. * (ej - ei) * numpy.dot(tdip, tdip))

class RQR(QR):
    '''Quadratic response for restricted RHF/RKS references.'''

    transition_dipole = transition_dipole
    oscillator_strength = oscillator_strength

    def _init_gxc(self):
        nvirt = int(numpy.count_nonzero(self.mo_occ == 0))
        shape = gxc_tensor_shape(
            self._manifold_n, self._manifold_m, nvirt)
        self._gxc = numpy.zeros(shape)
        if self.precompute_gxc:
            self._gxc_backend = EagerGxc(self._gxc)
        else:
            self._gxc_backend = LazyGxc()

    def kernel(self, *args, **kwargs):
        '''QR-stage setup: optionally precompute ``Gxc`` in memory.'''
        log = lib.logger.new_logger(self)
        log.info('QR kernel: manifolds ready (n=%d, m=%d states)',
                 len(self._manifold_n.e), len(self._manifold_m.e))
        if self.precompute_gxc:
            log.info('QR kernel: precomputing Gxc (%s)', self._gxc.shape)
            _precompute_gxc(
                self._scf, self._gxc,
                self._manifold_n.occ_idx, self._manifold_m.occ_idx)
        return self

    def get_2tdm(self, i, j):
        '''Excited-to-excited state transition density matrix (2TDM).

        Works for both TDA and RPA manifolds.

        Parameters
        ----------
        i : int
            0-based index into ``manifold_n``.
        j : int
            0-based index into ``manifold_m``.

        Returns
        -------
        tdm : ndarray
            Shape ``(nmo, nmo)``.
        '''
        self._sanity_check_2tdm(i, j)

        log = lib.logger.new_logger(self)
        log.info('  Computing C.')

        # TODO: cache these! The next two LOC are pure functions.
        C = _compute_c(self._scf)
        nocc, nvirt = C.shape[:2]
        A, B = _get_ab_from_c(C, self.mo_energy)
        Lambda = numpy.block([[A,B],[B,A]])
        Delta = numpy.eye(2*nocc*nvirt)
        Delta[nocc*nvirt:] *= -1

        log.info('  LHS of Casida eq. fully determined. Computing Gxc term.')
        x1_g, y1_g = self._manifold_n.xy[i]
        x2_g, y2_g = self._manifold_m.xy[j]

        if self.response_type == 'tda':
            V = self._gxc_backend.contract_v(self._scf, x1_g, x2_g)
        else:
            V = self._gxc_backend.contract_v(self._scf, x1_g + y1_g, x2_g + y2_g)

        log.info('  Gxc done. Determining RHS of Casida eq.')
        # Making explicit these quantities WILL NOT BE USED later.
        # Instead, we will get XY aligned to MO orbital indexing
        del x1_g, y1_g, x2_g, y2_g

        e1, (x1, y1) = self._manifold_n(i)
        e2, (x2, y2) = self._manifold_m(j)

        tdm = _get_2tdm_diag_block(x1, x2, y1, y2)

        Pia, Qia = _get_pq(C, tdm, V, x1, x2, y1, y2)
        pqm = numpy.reshape((Pia, Qia), -1)

        log.info('  Solving Casida eq. with ω = ΩM - ΩN.')
        xym = numpy.linalg.solve(Lambda - (e2-e1)*Delta, -pqm)
        _x2, _y2 = numpy.reshape(xym, (2,nocc,nvirt))

        # Fill in off-diagonal blocks of 2TDM. Some flip the convention by
        # transposing. I think this follows PySCFs linear response convention.
        tdm[:nocc,nocc:] = _y2
        tdm[nocc:,:nocc] = _x2.T

        return tdm

