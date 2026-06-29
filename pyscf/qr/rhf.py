'''Quadratic response driver for restricted RHF/RKS references.'''

import numpy
from scipy.linalg import block_diag

from pyscf import lib

from pyscf.qr.hf import QR
from pyscf.qr.manifold import align_xy, gxc_tensor_shape, union_occ_idx


def _precompute_gxc(mf, G, mo_occ):
    '''Fill the 6-index :math:`g_\\text{xc}` buffer in place.'''
    raise NotImplementedError('Gxc precomputation is not implemented yet.')


def _compute_c(mf):
    '''Returns C matrix of shape (nocc, nvirt, nmo, nmo)

    TODO: eventually cache this as well just like Gxc.
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

    # fxc contribution
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
    nocc, nvirt = C.shape[:2]
    e_ia = mo_energy[nocc:] - mo_energy[:nocc,None]
    A = numpy.diag(e_ia.ravel()).reshape(nocc,nvirt,nocc,nvirt)
    B = numpy.zeros_like(A)
    A += C[:,:,nocc:,:nocc].transpose((0,1,3,2))
    B += C[:,:,:nocc,nocc:]
    return numpy.reshape((A,B), (2,nocc*nvirt,nocc*nvirt))


class LazyGxc:
    '''Compute ``g_xc`` new each time for each state pair.

    Avoids storing the 6-index tensor but repeats grid work per call.
    This is generally faster if you only need a few calculations.

    Notes
    -----
    TDA contracts against raw ``x`` vectors (see reference ``get_v``), not
    ``x + y``.  Pass ``y=0`` via :meth:`contract_v` callers for TDA.
    '''

    def contract_v(self, mf, xpy1, xpy2, mo_occ=None):
        '''Return the ``V_ia`` vector for one pair of transition densities.

        xpy represents the sum of X and Y.
        '''
        raise NotImplementedError('Ad hoc Gxc contraction is not implemented yet.')


class EagerGxc:
    '''Contract :math:`g_\\text{xc}` from a precomputed 6-index tensor in memory.

    ``G`` has shape ``(nocc, nvirt, nocc_n, nvirt_n, nocc_m, nvirt_m)`` and is
    filled in place by :meth:`RQR.kernel` when ``precompute_gxc=True``.  The
    tensor is never written to checkpoint files.
    '''

    def __init__(self, G):
        self.G = G

    def contract_v(self, mf, xpy1, xpy2, mo_occ=None):
        return numpy.einsum('iajbkc,jb,kc->ia', self.G, xpy1, xpy2, optimize=True)


class RQR(QR):
    '''Quadratic response for restricted RHF/RKS references.

    **Gxc:** cached in memory only when ``precompute_gxc=True``; never
    written to checkpoint files.
    '''

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
            _precompute_gxc(self._scf, self._gxc, self._manifold_n.occ_idx, self._manifold_m.occ_idx)
        return self

    def get_2tdm(self, i, j):
        '''Excited-to-excited state transition density matrix (2TDM).

        Parameters
        ----------
        i : int
            0-based index into ``manifold_n.e`` / ``manifold_n.xy``.
        j : int
            0-based index into ``manifold_m.e`` / ``manifold_m.xy``.

        Returns
        -------
        tdm : ndarray
            Shape ``(nmo, nmo)``.

        Raises
        ------
        IndexError
            When ``i`` or ``j`` is out of range.
        '''
        self._sanity_check_2tdm(i, j)

        # TODO: cache these! The next two LOC are pure functions.
        C = _compute_c(self._scf)
        A, B = _get_ab_from_c(C, self.mo_energy)

        # e_i = float(self._manifold_n.e[i])
        # e_j = float(self._manifold_m.e[j])

        x1_g, y1_g = self._manifold_n.xy[i]
        x2_g, y2_g = self._manifold_m.xy[j]

        V = self._gxc_backend.contract_v(self._scf, x1_g, x2_g)

        # TODO: get aligned X and Y from _manifold attribute
        # x1, y1 = align_xy(self._manifold_n, i)
        # x2, y2 = align_xy(self._manifold_m, j)

        # TODO: func to get diagonal blocks of 2TDM.
        # goo = -(x1@x4.T + y4@y1.T)
        # gvv = (y1.T@y4 + x4.T@x1)
        # Knm = block_diag(goo, gvv)

        # TODO: call function to compute PQ (RHS of casida eq.) 
        # Pia = np.einsum('iapq,pq->ia', C, Knm) + V
        # Qia = np.einsum('iapq,pq->ia', C, Knm.T) + V
        # Ha = np.einsum('iapq,ia->pq', C, y1) + np.einsum('iaqp,ia->pq', C, x1)
        # Hb = np.einsum('iapq,ia->pq', C, x4) + np.einsum('iaqp,ia->pq', C, y4)
        # oo = (slice(nocc),slice(nocc))
        # vv = (slice(nocc,None),slice(nocc,None))
        # Pia += x4@Ha[vv].T - Ha[oo].T@x4 + y1@Hb[vv].T - Hb[oo].T@y1
        # Qia += y4@Ha[vv] - Ha[oo]@y4 + x1@Hb[vv] - Hb[oo]@x1
        # pqm = np.reshape((Pia,Qia),-1)

        # Solve Casida's equation with ω = ΩM - ΩN
        # xym = np.linalg.solve(Λ - (e4-e1)*Δ, -pqm)
        # _x2, _y2 = np.reshape(xym, (2,nocc,nvirt))

        if self.response_type == 'tda':
            pass
        else:
            pass
