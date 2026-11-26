# Copyright 2021-2026 The pyCMF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Partitioned equation-of-motion MP2 (P-EOM-MP2) for the band edges of
k-point OBMP2.  EXPERIMENTAL.

The ionization potentials and electron affinities are eigenvalues of the
similarity-transformed Hamiltonian exp(-T2) H exp(T2) in the space of 1h and
2h1p (1p and 2p1h) configurations, with the first-order doubles amplitudes T2
of the reference and the doubles-doubles block replaced by differences of
orbital energies (Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995);
Lange and Berkelbach, J. Chem. Phys. 155, 081101 (2021)).  Because that
block is diagonal, the doubles are eliminated exactly, which leaves an
energy-dependent matrix in the space of the bands of one k-point,

    IP:   M_ik(w) = -L_ki - sum_D C[i,D] W[k,D] / (w - E_D),  E_D = e_b - e_i - e_j
    EA:   M_ac(w) =  L_ac + sum_D C[a,D] W[c,D] / (w - E_D),  E_D = e_a + e_b - e_j

where C contains bare integrals and W the integrals dressed with T2.  Compared
with the second-order self-energy of :mod:`pycmf.pbc.kqp` (bare integrals on
both sides), the dressing adds the leading ring (screening), ladder and
exchange terms of third order.

The matrix is tabulated on a frequency grid around the band energy, in one
pass over the three-index integrals, and the eigenvalue equation
mu(w) = w is solved on a spline.  Only the target k-point enters as an
external index, so that no array with three k-point indices is stored.

Symmetry.  By default the singles space is the degenerate level of the band
edge, the outer k-point sum runs over the k-points that are inequivalent under
the operations which leave the target k-point fixed, and mu(w) is the trace
of M(w) over the level divided by its degeneracy.  This neglects the coupling
of the band edge to the other bands of the same k-point (about 0.01 eV for
diamond).  ``symmetry=False`` with all bands is the full calculation.

Conventions: orbitals, orbital energies (amplitude denominators and E_D) from
the KOBMP2 object; Fock matrix of Hartree-Fock in that orbital basis for the
singles block and the couplings.  With HF orbitals
(``KOBMP2(mf).init_from_scf()``) this is the usual P-EOM-MP2.  The result
agrees with the EOM-KCCSD code of PySCF run with t1 = 0, t2 = MP2 amplitudes
and the same doubles-doubles block.
'''

import numpy
from pyscf.lib import logger

from pycmf.pbc import kintegrals, kqp


def _amps(l1, l2, e1, e2, nk):
    '''t[p,a,q,b] = conj[(pa|qb)/Nk] / (e1[p,a] + e2[q,b])'''
    naux = l1.shape[0]
    s1, s2 = l1.shape[1:], l2.shape[1:]
    g = numpy.dot(l1.reshape(naux, -1).T, l2.reshape(naux, -1)).reshape(s1 + s2)
    g *= 1. / nk
    return (g / (e1[:, :, None, None] + e2[None, None])).conj()


def _eri(l1, l2, nk):
    '''(pq|rs)/Nk as [p,q,r,s]'''
    naux = l1.shape[0]
    s1, s2 = l1.shape[1:], l2.shape[1:]
    return numpy.dot(l1.reshape(naux, -1).T, l2.reshape(naux, -1)).reshape(s1 + s2) * (1. / nk)


class _Setup:
    '''Orbitals, energies, Fock blocks and integral access shared by IP and EA.'''

    def __init__(self, mp):
        self.log = logger.new_logger(mp)
        sym = mp._sym
        self.nk = nk = sym.nkpts
        self.kconserv = mp._kconserv()
        self.mo = [numpy.asarray(c) for c in mp.mo_coeff]
        self.e = [numpy.asarray(e) for e in mp.mo_energy]
        nocc = [int(numpy.count_nonzero(o > 0)) for o in mp.mo_occ]
        nmo = [c.shape[1] for c in self.mo]
        self.occ, self.vir = mp._active(nocc, nmo)
        if mp.with_df is None:
            mp.with_df = kintegrals.as_gdf(mp._scf)
        ints = kintegrals.KDFInts(mp.with_df, mp.kpts, mp.max_memory, mp.verbose, mp.stdout)
        ints.single_precision = getattr(mp, 'single_precision', False)
        ints.update(self.mo, self.occ, self.vir)
        self.ints = ints
        self.co = [self.mo[k][:, self.occ[k]] for k in range(nk)]
        self.cv = [self.mo[k][:, self.vir[k]] for k in range(nk)]
        self.eo = [self.e[k][self.occ[k]] for k in range(nk)]
        self.ev = [self.e[k][self.vir[k]] for k in range(nk)]
        # Hartree-Fock Fock matrix in the orbital basis
        fock = getattr(mp, 'fock_hf', None)
        if fock is None:
            mf = mp._scf
            dm = mf.make_rdm1(self.mo, mp.mo_occ)
            fao = mf.get_fock(dm=dm)
            fock = [c.conj().T.dot(fao[k]).dot(c) for k, c in enumerate(self.mo)]
        self.foo = [numpy.asarray(f)[numpy.ix_(o, o)] for f, o in zip(fock, self.occ)]
        self.fov = [numpy.asarray(f)[numpy.ix_(o, v)] for f, o, v in zip(fock, self.occ, self.vir)]
        self.fvv = [numpy.asarray(f)[numpy.ix_(v, v)] for f, v in zip(fock, self.vir)]
        self._oo = {}
        ints.build_rows()

    def ov(self, k1, k2):
        # the integral object keeps the rows (in memory or on disk, possibly in
        # single precision); no second copy here
        return self.ints.ov(k1, k2)

    def eov(self, k1, k2):
        return self.eo[k1][:, None] - self.ev[k2][None, :]

    def oo(self, k1, k2):
        if (k1, k2) not in self._oo:
            self._oo[k1, k2] = self.ints.block(k1, k2, self.co[k1], self.co[k2])
        return self._oo[k1, k2]

    def vv_row(self, k):
        return [self.ints.block(k, k2, self.cv[k], self.cv[k2]) for k2 in range(self.nk)]

    def amps(self, kp, ka, kq, kb):
        '''t[p,a,q,b] for p@kp, a@ka, q@kq, b@kb'''
        return _amps(self.ov(kp, ka), self.ov(kq, kb), self.eov(kp, ka), self.eov(kq, kb), self.nk)

    def window(self, half):
        '''Largest frequency window around a band edge without poles.'''
        top = max(e.max() for e in self.eo if e.size)
        bottom = min(e.min() for e in self.ev if e.size)
        return min(half, .7 * (bottom - top))


def _accumulate(m, c, w, e_d, lam, sign):
    """m[s,t,g] += sign sum_D c[s,D] w[t,D] / (lam[g] - e_d[D])"""
    ns, nt = c.shape[0], w.shape[0]
    c = c.reshape(ns, -1)
    w = w.reshape(nt, -1)
    inv = 1. / (lam[:, None] - e_d.reshape(1, -1))                      # [g,D]
    for g in range(lam.size):
        m[:, :, g] += sign * numpy.dot(c * inv[g], w.T)


def _wedge(sym, kpt, symmetry):
    """Representatives and weights of the orbits of the k-points under the
    operations that leave k-point ``kpt`` fixed."""
    nk = sym.nkpts
    if not symmetry:
        return list(range(nk)), [1] * nk
    ops = [op for op in sym._ops if sym._k2opk[kpt, op] == kpt]
    done = numpy.zeros(nk, dtype=bool)
    reps, weights = [], []
    for k in range(nk):
        if done[k]:
            continue
        orbit = {k} | {int(sym._k2opk[k, op]) for op in ops}
        done[list(orbit)] = True
        reps.append(k)
        weights.append(len(orbit))
    return reps, weights


def _level(e, target, tol=1e-4):
    """Indices of the bands degenerate with band ``target``.  The tolerance
    is far above the splitting left by an incompletely converged mean field
    (a level that is cut makes the trace over it meaningless) and far below
    the distance between different levels."""
    return numpy.where(abs(e - e[target]) < tol)[0]


def _solve(m, lam, target, trace):
    """Root of mu(w) = w, where mu is the eigenvalue of m(w) that belongs to
    band ``target`` (or the trace of m over a degenerate level).  Returns
    (root, mu at the centre of the grid)."""
    ng = lam.size
    mu = numpy.empty(ng)
    for g in range(ng):
        if trace:
            mu[g] = numpy.trace(m[:, :, g]).real / m.shape[0]
        else:
            w, v = numpy.linalg.eig(m[:, :, g])
            mu[g] = w[numpy.argmax(abs(v[target]))].real
    i0 = ng // 2
    root = kqp._root(lam, mu - lam, mu[i0])
    return root, mu[i0]


def _singles(n_all, target, nbands, symmetry, e):
    """Bands of the singles space, position of the target in it, trace mode."""
    level = _level(e, target)
    if nbands is None:
        sset = level if symmetry else numpy.arange(n_all)
    else:
        sset = numpy.asarray(nbands, dtype=int).ravel()
    trace = bool(symmetry)
    if trace and not numpy.array_equal(numpy.sort(sset), level):
        raise ValueError('With symmetry=True the singles space must be the degenerate level of the band edge.')
    return sset, int(numpy.where(sset == target)[0][0]), trace


def ip(mp, kpt=0, nbands=None, symmetry=True, grid=None, setup=None):
    """P-EOM-MP2 ionization potential of the highest occupied band at one
    k-point (index in the full mesh).

    Args:
        nbands : positions (among the correlated occupied bands of the
            k-point) of the bands in the singles space.  Default: the
            degenerate level of the band edge with ``symmetry=True``, all
            occupied bands otherwise.
        symmetry : use the operations that leave the k-point fixed

    Returns:
        dict with 'e_qp' (quasiparticle energy, -IP), 'e_ref' (band energy of
        the reference) and 'e_center' (-mu at the reference energy)
    """
    s = setup if setup is not None else _Setup(mp)
    log, nk, kcon, K = s.log, s.nk, s.kconserv, int(kpt)
    cput0 = (logger.process_clock(), logger.perf_counter())
    no_k = len(s.occ[K])
    sset, target, trace = _singles(no_k, no_k - 1, nbands, symmetry, s.eo[K])
    n_s = len(sset)
    half = s.window(.4)
    if grid is None:
        grid = numpy.linspace(-half, half, 41)
    lam = -s.eo[K][sset[target]] + numpy.asarray(grid, dtype=float)
    m = numpy.zeros((n_s, n_s, lam.size), dtype=numpy.complex128)
    pad = numpy.zeros((n_s, no_k))
    pad[numpy.arange(n_s), sset] = 1.
    fov_k = s.fov[K][sset]

    # singles block: -L_ki, L_ki = f_ki + sum (2 g - g_x)[k,c,l,d] t[i,c,l,d]
    loo = s.foo[K][numpy.ix_(sset, sset)].astype(numpy.complex128)
    for kl in range(nk):
        for kc in range(nk):
            kd = kcon[K, kc, kl]
            g = 2 * _eri(s.ov(K, kc)[:, sset], s.ov(kl, kd), nk)
            g -= _eri(s.ov(K, kd)[:, sset], s.ov(kl, kc), nk).transpose(0, 3, 2, 1)
            t = s.amps(K, kc, kl, kd)[sset]
            loo += numpy.tensordot(g, t, axes=([1, 2, 3], [1, 2, 3]))
    m -= loo.T[:, :, None]

    def left(kp, kq):
        """C[i; p,q,d] and E_D[p,q,d] of the block p@kp, q@kq, d@kd"""
        kd = kcon[kp, K, kq]
        c = -2 * _eri(s.oo(kp, K)[:, :, sset], s.ov(kq, kd), nk).transpose(1, 0, 2, 3)
        c += _eri(s.oo(kq, K)[:, :, sset], s.ov(kp, kd), nk).transpose(1, 2, 0, 3)
        if kp == K:
            c += 2 * numpy.einsum('ip,qd->ipqd', pad, s.fov[kq])
        if kq == K:
            c -= numpy.einsum('iq,pd->ipqd', pad, s.fov[kp])
        e_d = s.ev[kd][None, None, :] - s.eo[kp][:, None, None] - s.eo[kq][None, :, None]
        return c, e_d

    reps, weights = _wedge(mp._sym, K, symmetry)
    for kb, weight in zip(reps, weights):
        vv = s.vv_row(kb)
        for ki in range(nk):
            kj = kcon[kb, ki, K]
            l_jb = s.ov(kj, kb)
            naux, no_j, nv_b = l_jb.shape
            no_i = len(s.occ[ki])
            if 0 in (no_i, no_j, nv_b):
                continue
            n_q = numpy.zeros((naux, no_j, nv_b), dtype=numpy.complex128)
            w = numpy.zeros((n_s, nv_b, no_i, no_j), dtype=numpy.complex128)
            wx = numpy.zeros((n_s, nv_b, no_i, no_j), dtype=numpy.complex128)
            for kl in range(nk):
                kd = kcon[kl, kb, kj]
                l_ld = s.ov(kl, kd)
                if 0 in l_ld.shape:
                    continue
                t = s.amps(kl, kd, kj, kb)                              # [l,d,j,b]
                tx = s.amps(kl, kb, kj, kd).transpose(0, 3, 2, 1)       # t(lb,jd) as [l,d,j,b]
                n_q += numpy.tensordot(l_ld, 2 * t - tx, axes=([1, 2], [0, 1]))
                # -(li|kd) t(ld,jb)
                g = _eri(s.oo(kl, ki), s.ov(K, kd)[:, sset], nk)        # [l,i,k,d]
                w -= numpy.tensordot(g, t, axes=([0, 3], [0, 1])).transpose(1, 3, 0, 2)
                wx -= numpy.tensordot(g, tx, axes=([0, 3], [0, 1])).transpose(1, 3, 0, 2)
            for kd in range(nk):
                kc = kcon[kb, kd, K]
                if 0 in (len(s.vir[kd]), len(s.vir[kc])):
                    continue
                t = s.amps(kj, kd, ki, kc)                              # [j,d,i,c]
                x = numpy.tensordot(s.ov(K, kc)[:, sset], t, axes=([2], [3]))     # [Q,k,j,d,i]
                w += numpy.tensordot(vv[kd], x, axes=([0, 2], [0, 3])).transpose(1, 0, 3, 2) * (1. / nk)
            w += _eri(s.oo(ki, K)[:, :, sset], l_jb, nk).transpose(1, 3, 0, 2).conj()
            w += numpy.tensordot(s.oo(K, ki)[:, sset], n_q, axes=([0], [0])).transpose(0, 3, 1, 2) * (1. / nk)
            w += numpy.tensordot(fov_k, s.amps(ki, K, kj, kb), axes=([1], [1])).transpose(0, 3, 1, 2)
            c, e_d = left(ki, kj)
            _accumulate(m, c, w.transpose(0, 2, 3, 1), e_d, lam, -float(weight))
            # the exchange partner of this block: W[k,b,j,i] of the block (kj, ki)
            c, e_d = left(kj, ki)
            _accumulate(m, c, wx.transpose(0, 3, 2, 1), e_d, lam, -float(weight))
        vv = None
    root, mu0 = _solve(m, lam, target, trace)
    log.timer('P-EOM-MP2 IP', *cput0)
    return {'e_qp': -root, 'e_ref': s.eo[K][sset[target]], 'e_center': -mu0, 'lambda': lam, 'm': m}


def ea(mp, kpt=0, nbands=None, symmetry=True, grid=None, setup=None):
    """P-EOM-MP2 electron affinity of the lowest virtual band at one k-point.

    Args:
        nbands : positions (among the correlated virtual bands of the k-point)
            of the bands in the singles space; default as for :func:`ip`.
    """
    s = setup if setup is not None else _Setup(mp)
    log, nk, kcon, K = s.log, s.nk, s.kconserv, int(kpt)
    cput0 = (logger.process_clock(), logger.perf_counter())
    nv_k = len(s.vir[K])
    sel, target, trace = _singles(nv_k, 0, nbands, symmetry, s.ev[K])
    n_s = len(sel)
    half = s.window(.4)
    if grid is None:
        grid = numpy.linspace(-half, half, 41)
    lam = s.ev[K][sel[target]] + numpy.asarray(grid, dtype=float)
    m = numpy.zeros((n_s, n_s, lam.size), dtype=numpy.complex128)
    pad = numpy.zeros((n_s, nv_k))
    pad[numpy.arange(n_s), sel] = 1.
    fov_k = s.fov[K][:, sel]
    vv_k = [x[:, sel] for x in s.vv_row(K)]                             # L_ac(K, k2), a in S
    vv_to_k = [s.ints.block(k1, K, s.cv[k1], s.cv[K][:, sel]) for k1 in range(nk)]   # L_ac(k1, K), c in S

    # singles block: L_ac = f_ac - sum (2 g - g_x)[k,c,l,d] t[k,a,l,d]
    lvv = s.fvv[K][numpy.ix_(sel, sel)].astype(numpy.complex128)
    for kk in range(nk):
        for kl in range(nk):
            kd = kcon[kk, K, kl]
            g = 2 * _eri(s.ov(kk, K)[:, :, sel], s.ov(kl, kd), nk)
            g -= _eri(s.ov(kk, kd), s.ov(kl, K)[:, :, sel], nk).transpose(0, 3, 2, 1)
            t = s.amps(kk, K, kl, kd)[:, sel]
            lvv -= numpy.tensordot(t, g, axes=([0, 2, 3], [0, 2, 3]))
    m += lvv[:, :, None]

    def left(kj, ka):
        """C[a; j,c,d] and E_D[j,c,d] of the block j@kj, c@ka, d@kb"""
        kb = kcon[K, ka, kj]
        c = 2 * _eri(vv_k[ka], s.ov(kj, kb), nk).transpose(0, 2, 1, 3)
        c -= _eri(vv_k[kb], s.ov(kj, ka), nk).transpose(0, 2, 3, 1)
        if ka == K:
            c += 2 * numpy.einsum('ac,jd->ajcd', pad, s.fov[kj])
        if ka == kj:
            c -= numpy.einsum('ad,jc->ajcd', pad, s.fov[kj])
        e_d = s.ev[ka][None, :, None] + s.ev[kb][None, None, :] - s.eo[kj][:, None, None]
        return c, e_d

    reps, weights = _wedge(mp._sym, K, symmetry)
    for ka, weight in zip(reps, weights):
        vv = s.vv_row(ka)
        for kj in range(nk):
            kb = kcon[K, ka, kj]
            l_jb = s.ov(kj, kb)
            naux, no_j, nv_b = l_jb.shape
            nv_a = len(s.vir[ka])
            if 0 in (no_j, nv_b, nv_a):
                continue
            n_q = numpy.zeros((naux, no_j, nv_b), dtype=numpy.complex128)
            w = numpy.zeros((nv_a, nv_b, n_s, no_j), dtype=numpy.complex128)
            wx = numpy.zeros((nv_a, nv_b, n_s, no_j), dtype=numpy.complex128)
            for kl in range(nk):
                kd = kcon[kl, kb, kj]
                l_ld = s.ov(kl, kd)
                if 0 in l_ld.shape:
                    continue
                t = s.amps(kl, kd, kj, kb)                              # [l,d,j,b]
                tx = s.amps(kl, kb, kj, kd).transpose(0, 3, 2, 1)
                n_q += numpy.tensordot(l_ld, 2 * t - tx, axes=([1, 2], [0, 1]))
                # -(ad|lc) t(ld,jb)
                g = _eri(vv[kd], s.ov(kl, K)[:, :, sel], nk)            # [a,d,l,c]
                w -= numpy.tensordot(g, t, axes=([1, 2], [1, 0])).transpose(0, 3, 1, 2)
                wx -= numpy.tensordot(g, tx, axes=([1, 2], [1, 0])).transpose(0, 3, 1, 2)
            for kl in range(nk):
                kk = kcon[kb, kl, ka]
                if 0 in (len(s.occ[kl]), len(s.occ[kk])):
                    continue
                t = s.amps(kl, kb, kk, ka)                              # [l,b,k,a]
                g = _eri(s.oo(kl, kj), s.ov(kk, K)[:, :, sel], nk)      # [l,j,k,c]
                w += numpy.tensordot(g, t, axes=([0, 2], [0, 2])).transpose(3, 2, 1, 0)
            w += _eri(vv_k[ka], l_jb, nk).transpose(1, 3, 0, 2).conj()
            w += numpy.tensordot(vv_to_k[ka], n_q, axes=([0], [0])).transpose(0, 3, 1, 2) * (1. / nk)
            w -= numpy.tensordot(fov_k, s.amps(K, ka, kj, kb), axes=([0], [0])).transpose(1, 3, 0, 2)
            c, e_d = left(kj, ka)
            _accumulate(m, c, w.transpose(2, 3, 0, 1), e_d, lam, float(weight))
            # the exchange partner: W[b,a,c,j] of the block (kj, kb)
            c, e_d = left(kj, kb)
            _accumulate(m, c, wx.transpose(2, 3, 1, 0), e_d, lam, float(weight))
        vv = None
    root, mu0 = _solve(m, lam, target, trace)
    log.timer('P-EOM-MP2 EA', *cput0)
    return {'e_qp': root, 'e_ref': s.ev[K][sel[target]], 'e_center': mu0, 'lambda': lam, 'm': m}


def gap(mp, kpt_v=0, kpts_c=None, symmetry=True):
    """P-EOM-MP2 gaps from the highest occupied band at ``kpt_v`` to the
    lowest virtual band at each k-point of ``kpts_c`` (default: ``kpt_v`` and
    the k-point of the lowest virtual band of the reference).

    Returns:
        dict with 'ip' (quasiparticle energy of the valence band edge), 'ea'
        (dict k-point index -> quasiparticle energy) and 'gap' (dict).
    """
    s = _Setup(mp)
    if kpts_c is None:
        kmin = int(numpy.argmin([e[0] if e.size else numpy.inf for e in s.ev]))
        kpts_c = sorted({int(kpt_v), kmin})
    r_ip = ip(mp, kpt_v, symmetry=symmetry, setup=s)
    out = {'ip': r_ip['e_qp'], 'ea': {}, 'gap': {}}
    for k in kpts_c:
        r = ea(mp, k, symmetry=symmetry, setup=s)
        out['ea'][int(k)] = r['e_qp']
        out['gap'][int(k)] = r['e_qp'] - r_ip['e_qp']
    return out
