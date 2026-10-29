'''Quadratic response driver for restricted RHF/RKS references.'''

import numpy
from scipy.linalg import block_diag

from pyscf import lib, scf
from pyscf import ao2mo
from pyscf.dft.gen_grid import BLKSIZE

from pyscf.qr.hf import QR
from pyscf.qr.dipole import compute_dipole_mo
from pyscf.qr.intermediates import CasidaIntermediates
from pyscf.qr.manifold import gxc_tensor_shape


def _precompute_gxc(mf, G, occ_idx_n, occ_idx_m):
    '''Fill the 6-index :math:`g_\\text{xc}` buffer in place.'''
    if not isinstance(mf, scf.hf.KohnShamDFT):
        return G

    # TODO: create logger from QR object instead as verbose option may be overwritten.
    log = lib.logger.new_logger(mf)

    # needed for make_rho
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    mol = mf.mol

    occ_idx_n = numpy.asarray(occ_idx_n)
    occ_idx_m = numpy.asarray(occ_idx_m)
    assert mo_coeff.dtype == numpy.float64

    nao = mo_coeff.shape[0]
    occidx = numpy.where(mo_occ > 0)[0]
    viridx = numpy.where(mo_occ == 0)[0]
    nocc = len(occidx)
    nvir = len(viridx)

    nmo = nocc + nvir

    # TODO: assert occ_idx_n/m is a subset of occidx

    nocc_n = len(occ_idx_n)
    nocc_m = len(occ_idx_m)

    assert G.shape == (nocc, nvir, nocc_n, nvir, nocc_m, nvir)

    # fxc contribution
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)

    xctype = ni._xc_type(mf.xc)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]

    mem_now = lib.current_memory()[-1]
    max_memory = max(2000, mf.max_memory * .9 - mem_now)
    mf.grids.build()

    # yapf: disable
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords in ni.block_loop(
                mol, mf.grids, nao, ao_deriv, max_memory // (nocc * nvir)):
            rho = make_rho(0, ao, mask, xctype)
            gxc = ni.eval_xc_eff(mf.xc, rho, deriv=3, xctype=xctype)[3]

            rho = numpy.einsum('rp,pi->ri', ao, mo_coeff)
            rho_xv = numpy.einsum('rp,rq->rpq', rho, rho[:, viridx])
            rho_ov = rho_xv[:, occidx]
            rho_man1_ov = rho_xv[:, occ_idx_n]
            rho_man2_ov = rho_xv[:, occ_idx_m]

            wgxc = gxc[0, 0, 0] * weight
            w_ov = numpy.einsum('rkc,r->rkc', rho_man2_ov, wgxc, optimize=True)
            w_ovov = numpy.einsum('rjb,rkc->rjbkc', rho_man1_ov, w_ov, optimize=True)
            iajbkc = numpy.einsum( 'ria,rjbkc->iajbkc', rho_ov, w_ovov, optimize=True) * 4
            G += iajbkc
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords in ni.block_loop(
                mol, mf.grids, nao, ao_deriv, max_memory // (nocc * nvir)):
            rho = make_rho(0, ao, mask, xctype)
            gxc = ni.eval_xc_eff(mf.xc, rho, deriv=3, xctype=xctype)[3]

            rho = numpy.einsum('xrp,pi->xri', ao, mo_coeff)
            rho_xx = numpy.einsum('xrp,rq->xrpq', rho, rho[0])
            rho_xx[1:4] += numpy.einsum('rp,xrq->xrpq', rho[0], rho[1:4])
            rho_ov = rho_xx[:, :, occidx[:, None], viridx]
            rho_man1_ov = rho_xx[:, :, occ_idx_n[:, None], viridx]
            rho_man2_ov = rho_xx[:, :, occ_idx_m[:, None], viridx]

            wgxc = gxc * weight
            w_ov = numpy.einsum('xria,xyzr->yzria', rho_man2_ov, wgxc, optimize=True)
            iajb = numpy.einsum('xria,xyrjb->yriajb', rho_man1_ov, w_ov, optimize=True)
            iajbkc = numpy.einsum( 'xria,xrjbkc->iajbkc', rho_ov, iajb, optimize=True) * 4
            G += iajbkc
    else:
        raise NotImplementedError(f'xctype = {xctype}')
    # yapf: enable

    return G


def _get_2tdm_diag_block(x1, x2, y1, y2):
    '''Return 2TDM where off-diagonal blocks are 0.'''
    goo = -x1 @ x2.T
    gvv = x2.T @ x1
    if isinstance(y1, numpy.ndarray):
        goo -= y2 @ y1.T
        gvv += y1.T @ y2
    return block_diag(goo, gvv)


def _get_pq(C, Knm, V, x1, x2, y1, y2):
    '''Create |P,Q>: RHS of casida-like eq. for QR'''
    is_tda = not isinstance(y1, numpy.ndarray)
    nocc, nvirt = C.shape[:2]
    nmo = C.shape[2]
    oo = (slice(nocc), slice(nocc))
    vv = (slice(nocc, None), slice(nocc, None))

    # Flatten the pair indices so these contractions can use BLAS GEMV.
    # C is stored as (ia, pq), while the transposed Ha/Hb contractions below
    # require the corresponding pq transpose after the matrix product.
    Cflat = C.reshape(nocc * nvirt, nmo * nmo)
    Pia = (Cflat @ Knm.ravel()).reshape(nocc, nvirt) + V
    Qia = (Cflat @ Knm.T.ravel()).reshape(nocc, nvirt) + V

    Ha = (x1.ravel() @ Cflat).reshape(nmo, nmo).T
    Hb = (x2.ravel() @ Cflat).reshape(nmo, nmo)
    if not is_tda:
        Ha += (y1.ravel() @ Cflat).reshape(nmo, nmo)
        Hb += (y2.ravel() @ Cflat).reshape(nmo, nmo).T

    Pia += x2 @ Ha[vv].T - Ha[oo].T @ x2
    Qia += x1 @ Hb[vv] - Hb[oo] @ x1

    if is_tda:
        return Pia, Qia

    Pia += y1 @ Hb[vv].T - Hb[oo].T @ y1
    Qia += y2 @ Ha[vv] - Ha[oo] @ y2
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
    log = lib.logger.new_logger(mf)

    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ == 2)[0]
    viridx = numpy.where(mo_occ == 0)[0]
    orbv = mo_coeff[:, viridx]
    orbo = mo_coeff[:, occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = numpy.hstack((orbo, orbv))

    C = numpy.zeros((nocc, nvir, nmo, nmo))

    def add_hf_(c1, hyb=1):
        eri_shape = (nocc, nmo, nmo, nmo)
        eri_gib = numpy.prod(eri_shape) * mo_coeff.dtype.itemsize / 1024**3
        log.info(
            'QR memory: before full four-index MO ERI transformation; '
            'C shape=%s (%.2f GiB), eri_mo shape=%s (%.2f GiB)',
            c1.shape, c1.nbytes / 1024**3, eri_shape, eri_gib,
        )
        eri_mo = ao2mo.general(mol, [orbo, mo, mo, mo], compact=False)
        eri_mo = eri_mo.reshape(*eri_shape)
        log.info(
            'QR memory: after full four-index MO ERI transformation; '
            'eri_mo shape=%s (%.2f GiB)',
            eri_mo.shape, eri_mo.nbytes / 1024**3,
        )
        c1 += numpy.einsum('iapq->iapq', eri_mo[:nocc, nocc:]) * 2
        log.info('QR memory: after Coulomb contribution to C')
        c1 -= numpy.einsum('iqpa->iapq', eri_mo[:nocc, :, :, nocc:]) * hyb
        log.info('QR memory: after exchange contribution to C')

    if not isinstance(mf, scf.hf.KohnShamDFT):
        add_hf_(C)
        return C

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    add_hf_(C, hyb)
    if omega != 0:  # For RSH
        with mol.with_range_coulomb(omega):
            eri_mo = ao2mo.general(mol, [orbo, mo, mo, orbv], compact=False)
            eri_mo = eri_mo.reshape(nocc, nmo, nmo, nvir)
            k_fac = alpha - hyb
            C -= numpy.einsum('iqpa->iapq', eri_mo) * k_fac

    xctype = ni._xc_type(mf.xc)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]
    mem_now = lib.current_memory()[0]
    ## TODO: update memory usage
    max_memory = max(2000, mf.max_memory * .8 - mem_now)

    if xctype == 'GGA':
        ao_deriv = 1
        mf.grids.build()
        # At the largest point in this loop, rho_xx and w_xx coexist.  PySCF's
        # default block estimate only budgets the AO buffer, so it can select
        # a block that is far too large for these O(nmo**2) intermediates.
        gga_bytes_per_grid = C.dtype.itemsize * (
            4 * nao +                  # AO buffer
            4 + 2 * 16 +               # density, fxc, and weighted fxc
            4 * nmo +                  # MO values on the grid
            8 * nmo**2                 # rho_xx and w_xx
        )
        # C is resident and counted in mem_now.  Reserve one C-sized buffer
        # for the in-place-scaled ``iapq`` result before C += iapq.
        available = (mf.max_memory * .8 - mem_now) * 1e6 - C.nbytes
        block_units = int(available // gga_bytes_per_grid // BLKSIZE)
        if block_units < 4:
            raise MemoryError(
                'Insufficient memory for the minimum GGA QR grid block: '
                f'need at least {4 * BLKSIZE * gga_bytes_per_grid / 1e6:.0f} '
                f'MB of workspace after allocating C and iapq'
            )
        max_units = max(4, (len(mf.grids.coords) + BLKSIZE - 1) // BLKSIZE)
        blksize = min(block_units, max_units) * BLKSIZE
        nblocks = (len(mf.grids.coords) + blksize - 1) // blksize
        log.info(
            'QR memory: GGA XC block size=%d grid points '
            '(estimated workspace %.2f GiB)',
            blksize, blksize * gga_bytes_per_grid / 1024**3,
        )
        log.info('QR memory: GGA XC integration will process %d blocks',
                 nblocks)
        for block, (ao, mask, weight, coords) in enumerate(
                ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory,
                              blksize=blksize)):
            log.info(
                'QR memory: before GGA XC block %d; ao shape=%s, '
                'C shape=%s (%.2f GiB)',
                block, ao.shape, C.shape, C.nbytes / 1024**3,
            )
            rho = make_rho(0, ao, mask, xctype)

            fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
            wfxc = fxc * weight

            rho = numpy.einsum('xrp,pi->xri', ao, mo)
            rho_xx = numpy.einsum('xrp,rq->xrpq', rho, rho[0])
            rho_xx[1:4] += numpy.einsum('rp,xrq->xrpq', rho[0], rho[1:4])
            rho_xx = numpy.transpose(rho_xx, (1, 2, 3, 0))
            rho_ov = rho_xx[:, :nocc, nocc:]

            w_xx = numpy.einsum('rpqx,xyr->ypqr', rho_xx, wfxc, optimize=True)
            iapq = numpy.einsum('xpqr,riax->iapq', w_xx, rho_ov,
                                optimize=True)
            iapq *= 2
            C += iapq
            log.info(
                'QR memory: after GGA XC block %d; '
                'C shape=%s (%.2f GiB), iapq shape=%s (%.2f GiB)',
                block, C.shape, C.nbytes / 1024**3,
                iapq.shape, iapq.nbytes / 1024**3,
            )
            # These names otherwise stay bound until their replacements in
            # the next iteration are assigned.  That lets two grid blocks'
            # rho_xx/w_xx buffers coexist during the next allocation.
            del rho, fxc, wfxc, rho_ov, rho_xx, w_xx, iapq
    elif xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao,
                                                      ao_deriv, max_memory):
            rho = make_rho(0, ao, mask, xctype)
            fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
            wfxc = fxc[0, 0] * weight

            rho = numpy.einsum('rp,pi->ri', ao, mo)
            rho_xx = numpy.einsum('rp,rq->rpq', rho, rho)
            rho_ov = rho_xx[:, :nocc, nocc:]
            w_ov = numpy.einsum('ria,r->ria', rho_ov, wfxc)
            iapq = numpy.einsum('rpq,ria->iapq', rho_xx, w_ov)
            iapq *= 2
            C += iapq
    else:
        raise NotImplementedError(f'xctype = {xctype}')

    return C


def _get_ab_from_c(C, mo_energy):
    '''Returns the regular A,B matrices given C from :meth:``_compute_c``'''
    nocc, nvirt = C.shape[:2]
    e_ia = mo_energy[nocc:] - mo_energy[:nocc, None]
    A = numpy.diag(e_ia.ravel()).reshape(nocc, nvirt, nocc, nvirt)
    B = numpy.zeros_like(A)
    A += C[:, :, nocc:, :nocc].transpose((0, 1, 3, 2))
    B += C[:, :, :nocc, nocc:]
    return numpy.reshape((A, B), (2, nocc * nvirt, nocc * nvirt))


class Gxc:
    '''Contract the :math:`g_\\text{xc}` kernel against two excited states.

    A single backend supports both evaluation strategies:

    * **Eager** – when ``precompute_gxc`` is requested, a 6-index tensor of shape
      ``(nocc, nvirt, nocc_n, nvirt_n, nocc_m, nvirt_m)`` is allocated by
      :meth:`precompute` and filled in place.  Each
      :meth:`contract_v` call is then a cheap tensor contraction.  The tensor
      is never written to checkpoint files.
    * **Lazy** – otherwise the grid work is repeated on every
      :meth:`contract_v` call, avoiding the memory cost of the 6-index tensor.
      Generally faster when only a few calls are needed.

    The active occupied indices for each manifold, ``occ_idx_n`` and
    ``occ_idx_m``, are taken from the :class:`QR` instance at initialization.
    When either manifold uses frozen orbitals the last two pairs of eager
    indices differ, which saves space/computation and means no padding is
    required from excitation vectors before :meth:`contract_v`.
    '''

    def __init__(self, qr, precompute_gxc=False):
        self.occ_idx_n = numpy.asarray(qr._manifold_n.occ_idx)
        self.occ_idx_m = numpy.asarray(qr._manifold_m.occ_idx)

        self.precompute_gxc = bool(precompute_gxc)
        self.G = None
        if precompute_gxc:
            nvirt = int(numpy.count_nonzero(qr.mo_occ == 0))
            self._gxc_shape = gxc_tensor_shape(qr._manifold_n, qr._manifold_m,
                                               nvirt)
        else:
            self._gxc_shape = None

    def precompute(self, mf):
        '''Fill the in-memory 6-index tensor (eager mode only).

        No-op unless the backend was constructed with ``precompute_gxc=True``.
        '''
        if not self.precompute_gxc:
            return self
        if self.G is None:
            self.G = numpy.zeros(self._gxc_shape)
            _precompute_gxc(mf, self.G, self.occ_idx_n, self.occ_idx_m)
        return self

    def contract_v(self, mf, xpy1, xpy2, mo_occ=None):
        '''Return the ``V_ia`` vector given two excited states.

        ``xpy1`` represents the sum of ``x1`` and ``y1``: the first excitation.
        In eager mode this contracts the precomputed tensor; otherwise the
        grid work is repeated here.

        The shape of ``xpy`` should always be ``(nocc, nvirt)`` regardless of
        if there are frozen orbital indices.

        Notes
        -----
        The lazy evaluation here does not remove frozen orbitals. The frozen
        orbital indices are zeros and when contracted waste computation. The
        contraction from xy to grid space may be faster if xy was sliced. Since
        this contraction isn't the bottleneck, I'm not going to try to fix it.
        '''
        if self.precompute_gxc:
            if self.G is None:
                raise RuntimeError(
                    'QR.kernel() must be called before contract_v when '
                    'precompute_gxc=True')
            # Optimized for frozen orbitals
            xpy1_ = xpy1[self.occ_idx_n]
            xpy2_ = xpy2[self.occ_idx_m]
            return numpy.einsum('iajbkc,jb,kc->ia',
                                self.G,
                                xpy1_,
                                xpy2_,
                                optimize=True)

        if mo_occ is None: mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff

        assert mo_coeff.dtype == numpy.float64

        mol = mf.mol
        nao, nmo = mo_coeff.shape
        occidx = numpy.where(mo_occ == 2)[0]
        viridx = numpy.where(mo_occ == 0)[0]
        orbv = mo_coeff[:, viridx]
        orbo = mo_coeff[:, occidx]
        nvir = orbv.shape[1]
        nocc = orbo.shape[1]
        mo = numpy.hstack((orbo, orbv))

        G = numpy.zeros((nocc, nvir))

        if not isinstance(mf, scf.hf.KohnShamDFT):
            return G

        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
        xctype = ni._xc_type(mf.xc)

        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory * .8 - mem_now)

        if xctype == 'GGA':
            ao_deriv = 1
            mf.grids.build()
            # PySCF's default block estimate only accounts for the AO buffer.
            # This contraction also keeps rho_ov, w_ov, and iajb, whose sizes
            # are O(nocc * nvir * ngrids), alive at the same time.
            #
            # The factor 1.25 reserves room for einsum/libxc temporaries that
            # are not represented by a named array below.
            gga_bytes_per_grid = 1.25 * G.dtype.itemsize * (
                4 * nao +                 # AO buffer
                4 + 2 * 64 +              # rho, gxc, and wgxc
                4 * nocc + 4 * nvir +     # rho_o and rho_v
                8 +                       # xp1 and xp2
                24 * nocc * nvir          # rho_ov, w_ov, and iajb
            )
            available = (mf.max_memory * .8 - mem_now) * 1e6 - G.nbytes
            block_units = int(available // gga_bytes_per_grid // BLKSIZE)
            if block_units < 4:
                raise MemoryError(
                    'Insufficient memory for the minimum lazy GGA Gxc grid '
                    'block: need at least '
                    f'{4 * BLKSIZE * gga_bytes_per_grid / 1e6:.0f} MB '
                    'of workspace'
                )
            max_units = max(4, (len(mf.grids.coords) + BLKSIZE - 1) // BLKSIZE)
            blksize = min(block_units, max_units) * BLKSIZE
            nblocks = (len(mf.grids.coords) + blksize - 1) // blksize
            log = lib.logger.new_logger(mf)
            log.info(
                'QR memory: lazy GGA Gxc block size=%d grid points '
                '(estimated workspace %.2f GiB)',
                blksize, blksize * gga_bytes_per_grid / 1024**3,
            )
            log.info('QR memory: lazy GGA Gxc will process %d blocks',
                     nblocks)
            for ao, mask, weight, coords in ni.block_loop(
                    mol, mf.grids, nao, ao_deriv, max_memory,
                    blksize=blksize):
                rho = make_rho(0, ao, mask, xctype)
                gxc = ni.eval_xc_eff(mf.xc, rho, deriv=3, xctype=xctype)[3]

                rho_o = numpy.einsum('xrp,pi->xri', ao, orbo)
                rho_v = numpy.einsum('xrp,pi->xri', ao, orbv)
                rho_ov = numpy.einsum('xrp,rq->xrpq', rho_o, rho_v[0])
                rho_ov[1:4] += numpy.einsum('rp,xrq->xrpq', rho_o[0],
                                            rho_v[1:4])
                # This makes later contractions much faster
                rho_ov = numpy.transpose(rho_ov, (1, 2, 3, 0))

                # Convert excitation in MO basis to grid domain.
                xp1 = numpy.einsum('riax,ia->rx', rho_ov, xpy1)
                xp2 = numpy.einsum('riax,ia->rx', rho_ov, xpy2)

                # '...r,r...->...' looks like matrix multiplication. This is fast!
                wgxc = gxc * weight
                w_ov = numpy.einsum('riax,xyzr->yziar',
                                    rho_ov,
                                    wgxc,
                                    optimize=True)
                iajb = numpy.einsum('rx,xyiar->yiar', xp1, w_ov, optimize=True)
                iajbkc = numpy.einsum('xiar,rx->ia', iajb, xp2,
                                      optimize=True) * 4
                G += iajbkc
                # Ensure the previous grid block's large intermediates do
                # not coexist with the next block while its arrays are made.
                del rho, gxc, rho_o, rho_v, rho_ov, xp1, xp2, wgxc, w_ov, iajb, iajbkc
        elif xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords in ni.block_loop(
                    mol, mf.grids, nao, ao_deriv, max_memory):
                rho = make_rho(0, ao, mask, xctype)
                gxc = ni.eval_xc_eff(mf.xc, rho, deriv=3, xctype=xctype)[3]

                rho = numpy.einsum('rp,pi->ri', ao, mo)
                rho_xx = numpy.einsum('rp,rq->rpq', rho, rho)
                rho_ov = rho_xx[:, :nocc, nocc:]

                xp1 = numpy.einsum('ria,ia->r', rho_ov, xpy1)
                xp2 = numpy.einsum('ria,ia->r', rho_ov, xpy2)

                wgxc = gxc[0, 0, 0] * weight
                w_ov = numpy.einsum('ria,r->iar', rho_ov, wgxc, optimize=True)
                iajb = numpy.einsum('r,iar->iar', xp1, w_ov, optimize=True)
                iajbkc = numpy.einsum('iar,r->ia', iajb, xp2,
                                      optimize=True) * 4
                G += iajbkc
        else:
            raise NotImplementedError(f'xctype = {xctype}')

        return G


def transition_dipole(qrobj, tdm, dipole_mo=None):
    '''Returns transition dipole moment given excited-to-excited state 2TDM.

    Parameters
    ----------
    self : QR object
    tdm : ndarray
        Excited-to-excited state transition density matrix.
    dipole_mo : ndarray, optional
        Precomputed length-gauge dipole integrals in the MO basis.

    Returns
    -------
    tdip : ndarray
        Dipole (x,y,z) components.
    '''
    if dipole_mo is None:
        dipole_mo = compute_dipole_mo(qrobj.mol, qrobj.mo_coeff)
    # Factor of 2 for the beta-spin contribution (closed-shell reference).
    return numpy.einsum('xpq,pq->x', dipole_mo, tdm) * 2


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
    return float(2. / 3. * (ej - ei) * numpy.dot(tdip, tdip))


class RQR(QR):
    '''Quadratic response for restricted RHF/RKS references.'''

    transition_dipole = transition_dipole
    oscillator_strength = oscillator_strength

    def _init_gxc(self):
        self._gxc_backend = Gxc(self, precompute_gxc=self.precompute_gxc)

    def _build_intermediates(self):
        C = _compute_c(self._scf)
        A, B = _get_ab_from_c(C, self.mo_energy)
        return CasidaIntermediates.from_ab(C, A, B)

    def kernel(self, *args, **kwargs):
        '''QR-stage setup: optionally precompute ``Gxc`` in memory.'''
        log = lib.logger.new_logger(self)
        log.info('QR kernel: manifolds ready (n=%d, m=%d states)',
                 len(self._manifold_n.e), len(self._manifold_m.e))
        # Don't compute Gxc for Zero approximation. (TODO: fix code smell here)
        if self.precompute_gxc and self.approximation != 'Zero':
            log.info('QR kernel: precomputing Gxc (%s)',
                     self._gxc_backend._gxc_shape)
            self._gxc_backend.precompute(self._scf)
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

        if self.approximation not in ('Nascimento', 'Zero', 'Pseudo', None):
            raise ValueError('Invalid approxiamtion requested! Options are '
                             '\'Nascimento\', \'Zero\', \'Pseudo\', and None')
        log.info(f'  Using {self.approximation!r:s} approx.')

        log.info('  Computing diagonal blocks of 2TDM.')
        e1, (x1, y1) = self._manifold_n(i)
        e2, (x2, y2) = self._manifold_m(j)
        tdm = _get_2tdm_diag_block(x1, x2, y1, y2)
        if self.approximation == 'Nascimento':  # TODO: to_lower str if not None
            log.info('  Solving Casida eq. with Nascimento approx.')
            return tdm

        log.info('  Computing LHS intermediates.')
        inter = self.intermediates
        C, Lambda, Delta = inter.C, inter.Lambda, inter.Delta
        nocc, nvirt = inter.nocc, inter.nvirt

        log.info('  Computing Gxc term.')
        # This part is more likely to fail first compared to intermediates.
        if self.approximation == 'Zero':
            V = numpy.zeros((nocc, nvirt))
        else:
            if self.response_type == 'tda':
                V = self._gxc_backend.contract_v(self._scf, x1, x2)
            else:
                V = self._gxc_backend.contract_v(self._scf, x1 + y1, x2 + y2)

        log.info('  Computing P, Q.')
        Pia, Qia = _get_pq(C, tdm, V, x1, x2, y1, y2)
        pqm = numpy.reshape((Pia, Qia), -1)

        log.info('  Casida eq. knowns fully determined.')

        if self.approximation == 'Pseudo':
            log.info('  Solving Casida eq. with ω = 0.')
            xym = numpy.linalg.solve(Lambda, -pqm)
        else:
            log.info('  Solving Casida eq. with ω = ΩM - ΩN.')
            xym = numpy.linalg.solve(Lambda - (e2 - e1) * Delta, -pqm)
        _x2, _y2 = numpy.reshape(xym, (2, nocc, nvirt))

        # Fill in off-diagonal blocks of 2TDM. Some flip the convention by
        # transposing. I think this follows PySCFs linear response convention.
        tdm[:nocc, nocc:] = _y2
        tdm[nocc:, :nocc] = _x2.T

        return tdm

    def _transition_dipoles(self,
                            intermediate_states=None,
                            final_states=None,
                            dipole_mo=None):
        '''Compute selected excited-to-excited transition dipoles directly.

        This is the RIXS-specific counterpart to :meth:`get_2tdm`.  It keeps
        the pair-dependent Casida calculation, but contracts the diagonal and
        solved off-diagonal 2TDM blocks directly with ``dipole_mo``.
        '''
        def normalize_states(states, size, name):
            if states is None:
                return numpy.arange(size, dtype=int)
            states = numpy.atleast_1d(numpy.asarray(states))
            if states.ndim != 1:
                raise ValueError(f'{name} must be a one-dimensional array')
            if not numpy.issubdtype(states.dtype, numpy.integer):
                raise ValueError(f'{name} must contain integer state indices')
            states = states.astype(int, copy=False)
            if numpy.any(states < 0) or numpy.any(states >= size):
                raise IndexError(
                    f'{name} contains an index outside [0, {size})')
            return states

        n_states = normalize_states(
            intermediate_states,
            len(self._manifold_n.e),
            'intermediate_states',
        )
        f_states = normalize_states(
            final_states,
            len(self._manifold_m.e),
            'final_states',
        )

        dtype = numpy.result_type(self.mo_coeff.dtype, numpy.float64)
        if len(n_states) == 0 or len(f_states) == 0:
            if dipole_mo is not None:
                dtype = numpy.result_type(dtype, dipole_mo)
            return numpy.empty((3, len(f_states), len(n_states)),
                               dtype=dtype)

        if self.approximation not in ('Nascimento', 'Zero', 'Pseudo', None):
            raise ValueError('Invalid approxiamtion requested! Options are '
                             '\'Nascimento\', \'Zero\', \'Pseudo\', and None')

        if dipole_mo is None:
            dipole_mo = compute_dipole_mo(self.mol, self.mo_coeff)
        dipole_mo = numpy.asarray(dipole_mo)
        dtype = numpy.result_type(dtype, dipole_mo)

        is_rpa = self.response_type == 'rpa'
        if self.approximation == 'Nascimento':
            Cflat = None
            Coo = Cvv = None
            nocc = int(numpy.count_nonzero(self.mo_occ > 0))
            nvirt = int(numpy.count_nonzero(self.mo_occ == 0))
            Lambda = Delta = None
        else:
            inter = self.intermediates
            C = inter.C
            nocc, nvirt = inter.nocc, inter.nvirt
            nmo = C.shape[2]
            Cflat = C.reshape(nocc * nvirt, nmo * nmo)

            # Select the occupied-occupied and virtual-virtual entries from
            # Cflat once.  These are the only C entries needed by the
            # diagonal 2TDM blocks.
            occ = numpy.arange(nocc)
            virt = nocc + numpy.arange(nvirt)
            oo_idx = (occ[:, None] * nmo + occ[None, :]).ravel()
            vv_idx = (virt[:, None] * nmo + virt[None, :]).ravel()
            Coo = Cflat[:, oo_idx]
            Cvv = Cflat[:, vv_idx]
            Lambda, Delta = inter.Lambda, inter.Delta

        oo_slice = slice(nocc)
        vv_slice = slice(nocc, nocc + nvirt)
        dip_oo = dipole_mo[:, oo_slice, oo_slice]
        dip_ov = dipole_mo[:, oo_slice, vv_slice]
        dip_vo = dipole_mo[:, vv_slice, oo_slice]
        dip_vv = dipole_mo[:, vv_slice, vv_slice]

        aligned = {}

        def get_aligned(manifold, state):
            key = (id(manifold), int(state))
            if key not in aligned:
                aligned[key] = manifold(int(state))
            return aligned[key]

        n_data = {}
        f_data = {}
        for state in n_states:
            state = int(state)
            if state not in n_data:
                e, (x, y) = get_aligned(self._manifold_n, state)
                if Cflat is None:
                    ha = None
                else:
                    ha = (x.ravel() @ Cflat).reshape(nmo, nmo).T
                    if is_rpa:
                        ha += (y.ravel() @ Cflat).reshape(nmo, nmo)
                n_data[state] = (e, x, y, ha)
        for state in f_states:
            state = int(state)
            if state not in f_data:
                e, (x, y) = get_aligned(self._manifold_m, state)
                if Cflat is None:
                    hb = None
                else:
                    hb = (x.ravel() @ Cflat).reshape(nmo, nmo)
                    if is_rpa:
                        hb += (y.ravel() @ Cflat).reshape(nmo, nmo).T
                f_data[state] = (e, x, y, hb)

        dipoles = numpy.empty((3, len(f_states), len(n_states)), dtype=dtype)
        n_records = [n_data[int(state)] for state in n_states]
        f_records = [f_data[int(state)] for state in f_states]

        if self.approximation == 'Nascimento':
            for f_pos, (_, x2, y2, _) in enumerate(f_records):
                for n_pos, (_, x1, y1, _) in enumerate(n_records):
                    knm_oo = -(x1 @ x2.T)
                    knm_vv = x2.T @ x1
                    if is_rpa:
                        knm_oo -= y2 @ y1.T
                        knm_vv += y1.T @ y2
                    dipoles[:, f_pos, n_pos] = 2 * (
                        numpy.einsum('xij,ij->x', dip_oo, knm_oo,
                                     optimize=True)
                        + numpy.einsum('xab,ab->x', dip_vv, knm_vv,
                                       optimize=True)
                    )
            return dipoles

        # Keep the pair-dependent RHS buffers bounded while turning the C
        # contractions into matrix-matrix products.
        n_e = numpy.asarray([record[0] for record in n_records])
        f_e = numpy.asarray([record[0] for record in f_records])
        pair_f = numpy.repeat(numpy.arange(len(f_states)),
                              len(n_states))
        pair_n = numpy.tile(numpy.arange(len(n_states)),
                            len(f_states))
        pair_block_size = 32

        for block_start in range(0, len(pair_f), pair_block_size):
            block_stop = min(block_start + pair_block_size, len(pair_f))
            f_idx = pair_f[block_start:block_stop]
            n_idx = pair_n[block_start:block_stop]
            x1 = numpy.stack([n_records[idx][1] for idx in n_idx])
            x2 = numpy.stack([f_records[idx][1] for idx in f_idx])
            ha = numpy.stack([n_records[idx][3] for idx in n_idx])
            hb = numpy.stack([f_records[idx][3] for idx in f_idx])

            knm_oo = -numpy.matmul(x1, x2.transpose(0, 2, 1))
            knm_vv = numpy.matmul(x2.transpose(0, 2, 1), x1)
            if is_rpa:
                y1 = numpy.stack([n_records[idx][2] for idx in n_idx])
                y2 = numpy.stack([f_records[idx][2] for idx in f_idx])
                knm_oo -= numpy.matmul(y2, y1.transpose(0, 2, 1))
                knm_vv += numpy.matmul(y1.transpose(0, 2, 1), y2)

            tdip = (
                numpy.einsum('xij,bij->bx', dip_oo, knm_oo,
                             optimize=True)
                + numpy.einsum('xab,pab->px', dip_vv, knm_vv,
                               optimize=True)
            )

            if self.approximation == 'Zero':
                V = numpy.zeros((block_stop - block_start, nocc, nvirt))
            else:
                V = []
                for pair_pos in range(block_stop - block_start):
                    if is_rpa:
                        V.append(self._gxc_backend.contract_v(
                            self._scf,
                            x1[pair_pos] + y1[pair_pos],
                            x2[pair_pos] + y2[pair_pos],
                        ))
                    else:
                        V.append(self._gxc_backend.contract_v(
                            self._scf, x1[pair_pos], x2[pair_pos]))
                V = numpy.asarray(V)

            Pia = (
                knm_oo.reshape(len(f_idx), -1) @ Coo.T
                + knm_vv.reshape(len(f_idx), -1) @ Cvv.T
            ).reshape(len(f_idx), nocc, nvirt)
            Qia = (
                knm_oo.transpose(0, 2, 1).reshape(len(f_idx), -1) @ Coo.T
                + knm_vv.transpose(0, 2, 1).reshape(len(f_idx), -1)
                @ Cvv.T
            ).reshape(len(f_idx), nocc, nvirt)
            Pia += V
            Qia += V

            ha_oo = ha[:, oo_slice, oo_slice]
            ha_vv = ha[:, vv_slice, vv_slice]
            hb_oo = hb[:, oo_slice, oo_slice]
            hb_vv = hb[:, vv_slice, vv_slice]
            Pia += (
                numpy.matmul(x2, ha_vv.transpose(0, 2, 1))
                - numpy.matmul(ha_oo.transpose(0, 2, 1), x2)
            )
            Qia += numpy.matmul(x1, hb_vv) - numpy.matmul(hb_oo, x1)
            if is_rpa:
                Pia += (
                    numpy.matmul(y1, hb_vv.transpose(0, 2, 1))
                    - numpy.matmul(hb_oo.transpose(0, 2, 1), y1)
                )
                Qia += numpy.matmul(y2, ha_vv) - numpy.matmul(ha_oo, y2)

            pqm = numpy.concatenate((Pia, Qia), axis=1).reshape(
                len(f_idx), -1)
            for pair_pos, (f_pos, n_pos) in enumerate(zip(f_idx, n_idx)):
                if self.approximation == 'Pseudo':
                    xym = numpy.linalg.solve(Lambda, -pqm[pair_pos])
                else:
                    omega = f_e[f_pos] - n_e[n_pos]
                    xym = numpy.linalg.solve(
                        Lambda - omega * Delta,
                        -pqm[pair_pos],
                    )
                x_response, y_response = numpy.reshape(
                    xym, (2, nocc, nvirt))
                tdip_pair = tdip[pair_pos] + (
                    numpy.einsum('xia,ia->x', dip_ov, y_response,
                                 optimize=True)
                    + numpy.einsum('xai,ia->x', dip_vo, x_response,
                                   optimize=True)
                )
                dipoles[:, f_pos, n_pos] = 2 * tdip_pair

        return dipoles
