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
Correlation potential of OBMP2 with k-point sampling (closed shell).

The working equations are those of :mod:`pycmf.mol.bch`, spin-summed and
with crystal momentum conservation.  For the four k-points (ki, ka, kj, kb)
of a double excitation ia -> jb,

    g[i,a,j,b]    = (ia|jb) / Nk
    t[i,a,j,b]    = conj(g[i,a,j,b]) / (e_i - e_a + e_j - e_b - shift)
    tbar[i,a,j,b] = alpha t[i,a,j,b] - beta t[i,b,j,a]

with alpha = (c_ss + c_os)/2 and beta = c_ss/2 in the terms linear in the
amplitudes, and the squares of c_ss, c_os in the quadratic terms (the same
convention as the spin blocks of :mod:`pycmf.core.amplitudes`).

Organisation of the sums
------------------------
The potential is accumulated in two kinds of terms.  "j terms" carry their
free indices on the k-point kj, "b terms" on kb.  The terms that the original
formulation attaches to ki and ka are rewritten with the pair symmetry
t[i,a,j,b] = t[j,b,i,a], so that every term is of one of the two kinds.

* kj runs over the irreducible k-points only.  The j terms are then complete
  for these k-points, and the b terms are partial sums that are completed by
  the group average of :class:`pycmf.pbc.ksymm.KSymmetry`.
* For a pair (kj, ki) the loop over ka builds the integrals from three-index
  tensors, the amplitudes, and the intermediate

      N^Q[j,b](kb) = sum_{ki} sum_{ia} L^Q[i,a](ki,ka) tbar[i,a,j,b],

  which replaces all the integrals with a general orbital index:

      sum_{ia} tbar[i,a,j,b] (ia|pb) = sum_Q N^Q[j,b] L^Q[p,b].

No four-index quantity is stored beyond the amplitudes of one k-point
quadruple.  The loop over ki is distributed with MPI when available.
'''

import numpy
from pyscf.lib import logger

from pycmf.core import mpi


def _denominator(e_ia, e_jb, shift):
    return e_ia[:, :, None, None] + e_jb[None, None, :, :] - shift


def _ovov(l_ia, l_jb, scale):
    '''scale * sum_Q L[Q,i,a] L[Q,j,b]'''
    naux = l_ia.shape[0]
    shape = l_ia.shape[1:] + l_jb.shape[1:]
    if 0 in shape:
        return numpy.zeros(shape, dtype=numpy.complex128)
    g = numpy.dot(l_ia.reshape(naux, shape[0] * shape[1]).T, l_jb.reshape(naux, shape[2] * shape[3]))
    g *= scale
    return g.reshape(shape)


class _Sums:
    '''Accumulators of one evaluation of the correlation potential.'''

    def __init__(self, sym, nmo, occ, vir):
        nk = sym.nkpts
        no = [len(x) for x in occ]
        nv = [len(x) for x in vir]
        z = lambda *shape: numpy.zeros(shape, dtype=numpy.complex128)
        ibz = sym.ibz2bz
        # complete sums on the irreducible k-points
        self.cb = [z(nmo[k], no[k]) for k in ibz]      # [p,j]
        self.coo = [z(no[k], no[k]) for k in ibz]      # [l,j]
        self.y1 = [z(nv[k], no[k]) for k in ibz]       # [b,j]
        self.y4 = [z(no[k], no[k]) for k in ibz]       # [l,j]
        # partial sums on all k-points
        self.pa = [z(nv[k], nmo[k]) for k in range(nk)]  # [b,p]
        self.pvv = [z(nv[k], nv[k]) for k in range(nk)]  # [b,d]
        self.y5 = [z(nv[k], nv[k]) for k in range(nk)]   # [b,d]
        self.c0 = numpy.zeros(2)                         # first and second order

    _lists = ('cb', 'coo', 'y1', 'y4', 'pa', 'pvv', 'y5')

    def reduce(self):
        if mpi.comm is None:
            return self
        arrays = [self.c0]
        for name in self._lists:
            arrays += getattr(self, name)
        out = mpi.allreduce(arrays)
        self.c0 = out[0].real
        p = 1
        for name in self._lists:
            n = len(getattr(self, name))
            setattr(self, name, out[p:p + n])
            p += n
        return self


def _triple(sums, i_ibz, weight, kb, li_a, g, gx, e, fock_blocks, coef, nq, second, diag_fov):
    '''Terms of one k-point quadruple (ki, ka, kj, kb).

    g = (ia|jb)/Nk [i,a,j,b];  gx = (ib|ja)/Nk as [i,b,j,a] or None.

    All contractions are written as matrix products of reshaped arrays: for
    small blocks the overhead of a general einsum would dominate, for large
    ones every product is a single BLAS call.
    '''
    (a1, b1), (a2, b2) = coef
    no_i, nv_a, no_j, nv_b = g.shape
    if g.size == 0:
        return
    n_ia, n_jb = no_i * nv_a, no_j * nv_b
    t = g / e
    numpy.conj(t, out=t)
    if gx is None:
        tx = None
        tb1 = t * a1 if a1 != 1 else t
    else:
        tx = gx.transpose(0, 3, 2, 1) / e
        numpy.conj(tx, out=tx)
        tb1 = a1 * t - b1 * tx

    # ---- terms linear in the amplitudes
    sums.c0[0] -= 4 * weight * numpy.dot(tb1.ravel(), g.ravel()).real
    naux = li_a.shape[0]
    m = numpy.dot(li_a.reshape(naux, n_ia), tb1.reshape(n_ia, n_jb)).reshape(naux, no_j, nv_b)
    if kb in nq:
        nq[kb] += m
    else:
        nq[kb] = m
    if diag_fov is not None:
        # ki == ka: singles-like term f_ia tbar
        y = numpy.dot(diag_fov.ravel(), tb1.reshape(n_ia, n_jb))
        sums.y1[i_ibz] += 2 * y.reshape(no_j, nv_b).T
    if not second:
        return

    # ---- terms quadratic in the amplitudes; tbc = conj(tbar)
    if (a1, b1) == (a2, b2):
        tbc = tb1.conj()
    elif tx is None:
        tbc = a2 * t.conj()
    else:
        tbc = a2 * t.conj() - b2 * tx.conj()
    foo_i, foo_j, fvv_a, fvv_b = fock_blocks
    # y2[i,a,l,b] = sum_c f_vv(a)[c,a] tbc[i,c,l,b]
    y2 = numpy.dot(fvv_a.T, tbc.transpose(1, 0, 2, 3).reshape(nv_a, no_i * n_jb))
    y2 = y2.reshape(nv_a, no_i, no_j, nv_b).transpose(1, 0, 2, 3)
    # y3[i,a,l,b] = sum_k f_oo(i)[i,k] tbc[k,a,l,b]
    y3 = numpy.dot(foo_i, tbc.reshape(no_i, nv_a * n_jb)).reshape(g.shape)
    y3 -= y2
    y2 = None
    sums.c0[1] += 4 * weight * numpy.dot(y3.ravel(), t.ravel()).real
    # y3 now holds  f_oo(i) tbar - f_vv(a) tbar
    # xj[i,a,l,b] = sum_c tbc[i,a,l,c] f_vv(b)[c,b] - y3
    xj = numpy.dot(tbc.reshape(n_ia * no_j, nv_b), fvv_b).reshape(g.shape)
    xj -= y3
    # [l,j] <- sum_{iab} x[i,a,l,b] t[i,a,j,b]
    t_j = t.reshape(n_ia, no_j, nv_b).transpose(1, 0, 2).reshape(no_j, n_ia * nv_b)
    x_j = xj.reshape(n_ia, no_j, nv_b).transpose(1, 0, 2).reshape(no_j, n_ia * nv_b)
    sums.coo[i_ibz] += numpy.dot(x_j, t_j.T)
    xj = x_j = None
    # y3[i,a,j,d] += sum_k f_oo(j)[j,k] tbc[i,a,k,d]
    z = numpy.dot(foo_j, tbc.transpose(2, 0, 1, 3).reshape(no_j, n_ia * nv_b))
    y3 += z.reshape(no_j, no_i, nv_a, nv_b).transpose(1, 2, 0, 3)
    z = None
    # [b,d] <- sum_{iaj} t[i,a,j,b] x[i,a,j,d]
    t_b = t.reshape(n_ia * no_j, nv_b)
    sums.pvv[kb] += weight * numpy.dot(t_b.T, y3.reshape(n_ia * no_j, nv_b))
    y3 = None
    tb_j = tbc.reshape(n_ia, no_j, nv_b).transpose(1, 0, 2).reshape(no_j, n_ia * nv_b)
    sums.y4[i_ibz] += numpy.dot(tb_j, t_j.T)
    sums.y5[kb] += weight * numpy.dot(t_b.T, tbc.reshape(n_ia * no_j, nv_b))


def correlation_potential(ints, sym, kconserv, mo_energy, mo_coeff, s1e, fock, occ, vir,
                          c_os=1., c_ss=1., shift=0., bch_order=2, log=None):
    '''One-body correlation potential and constant of k-point OBMP2.

    Args:
        ints : :class:`pycmf.pbc.kintegrals.KDFInts`, updated for ``mo_coeff``
        sym : :class:`pycmf.pbc.ksymm.KSymmetry`
        kconserv : momentum conservation table of PySCF
        mo_energy, mo_coeff, s1e, fock : orbital energies, orbitals, overlap
            and HF Fock matrices (MO basis) for all k-points of the full zone,
            in the gauge of ``sym.expand``
        occ, vir : index arrays of the correlated occupied / virtual orbitals
        c_os, c_ss : scaling of opposite-spin and same-spin amplitudes
        shift : level shift of the denominators
        bch_order : 2 (default) or 1, as for molecules

    Returns:
        v : list over the irreducible k-points of Hermitian MO matrices
        c0 : constant (per unit cell)
        rdm : (r_occ, r_vir), lists over the irreducible k-points with the
            occupied and virtual blocks of the second-order density (None for
            bch_order = 1)
    '''
    if bch_order not in (1, 2):
        raise ValueError('bch_order must be 1 or 2')
    nk = sym.nkpts
    second = bch_order == 2
    coef = ((.5 * (c_ss + c_os), .5 * c_ss), (.5 * (c_ss**2 + c_os**2), .5 * c_ss**2))
    exchange = c_ss != 0
    nmo = [f.shape[0] for f in fock]
    e_o = [numpy.asarray(mo_energy[k])[occ[k]] for k in range(nk)]
    e_v = [numpy.asarray(mo_energy[k])[vir[k]] for k in range(nk)]
    f_oo = [fock[k][numpy.ix_(occ[k], occ[k])] for k in range(nk)]
    f_vv = [fock[k][numpy.ix_(vir[k], vir[k])] for k in range(nk)]
    f_ov = [fock[k][numpy.ix_(occ[k], vir[k])] for k in range(nk)]
    sums = _Sums(sym, nmo, occ, vir)
    if c_ss == 0 and c_os == 0:
        return [numpy.zeros((n, n), dtype=complex) for n in (nmo[k] for k in sym.ibz2bz)], 0., None

    my_ki = mpi.split(range(nk))
    t_amp = t_right = 0.
    for i_ibz, kj in enumerate(sym.ibz2bz):
        weight = float(sym.star[i_ibz])
        row_j = ints.row(kj)
        nq = {}
        t0 = logger.perf_counter()
        for ki in my_ki:
            row_i = ints.row(ki)
            done = numpy.zeros(nk, dtype=bool)
            for ka in range(nk):
                if done[ka]:
                    continue
                kb = kconserv[ki, ka, kj]
                la, lb = row_i[ka], row_j[kb]
                pair = exchange and kb != ka
                done[ka] = True
                if pair:
                    done[kb] = True
                if la.shape[1] == 0 or lb.shape[1] == 0:
                    # no occupied orbital at ki or kj
                    continue
                g = _ovov(la, lb, 1. / nk)
                if not exchange:
                    gx = None
                elif pair:
                    lxa = row_i[kb]
                    gx = _ovov(lxa, row_j[ka], 1. / nk)
                else:
                    gx = g
                blocks = (f_oo[ki], f_oo[kj], f_vv[ka], f_vv[kb]) if second else None
                e = _denominator(e_o[ki][:, None] - e_v[ka], e_o[kj][:, None] - e_v[kb], shift)
                _triple(sums, i_ibz, weight, kb, la, g, gx, e,
                        blocks, coef, nq, second, f_ov[ki] if ki == ka else None)
                if pair:
                    # the exchange partner (ki, kb, kj, ka) shares the integrals
                    blocks = (f_oo[ki], f_oo[kj], f_vv[kb], f_vv[ka]) if second else None
                    e = _denominator(e_o[ki][:, None] - e_v[kb], e_o[kj][:, None] - e_v[ka], shift)
                    _triple(sums, i_ibz, weight, ka, lxa, gx, g, e,
                            blocks, coef, nq, second, f_ov[ki] if ki == kb else None)
        t_amp += logger.perf_counter() - t0

        # contraction with the integrals that carry a general orbital index
        t0 = logger.perf_counter()
        for kb, n in nq.items():
            l_og, l_gv = ints.right(kj, kb)
            naux, no_j, nv_b = n.shape
            # - 2 sum_{Q j} N^Q[j,b] L^Q[j,p]   -> [b,p] on kb   (partial sum)
            x = numpy.dot(n.reshape(naux * no_j, nv_b).T, l_og.reshape(naux * no_j, l_og.shape[2]))
            sums.pa[kb] -= (2. * weight / nk) * x
            # + 2 sum_{Q b} L^Q[p,b] N^Q[j,b]   -> [p,j] on kj
            nmo_j = l_gv.shape[1]
            x = numpy.dot(l_gv.transpose(1, 0, 2).reshape(nmo_j, naux * nv_b),
                          n.transpose(1, 0, 2).reshape(no_j, naux * nv_b).T)
            sums.cb[i_ibz] += (2. / nk) * x
        nq = None
        t_right += logger.perf_counter() - t0
    sums.reduce()
    if log is not None:
        log.debug1('  amplitudes and sums %.2f s, general-index contraction %.2f s (rank %d)',
                   t_amp, t_right, mpi.rank)

    # ---- assemble on the irreducible k-points
    c0 = sums.c0 / nk
    full = []
    y5full = []
    for k in range(nk):
        p = numpy.zeros((nmo[k], nmo[k]), dtype=numpy.complex128)
        p[vir[k]] += sums.pa[k]
        p[numpy.ix_(vir[k], vir[k])] += sums.pvv[k]
        full.append(p)
        y = numpy.zeros_like(p)
        y[numpy.ix_(vir[k], vir[k])] = sums.y5[k]
        y5full.append(y)
    c1, y5 = sym.symmetrize([full, y5full], mo_coeff, s1e)
    for i_ibz, k in enumerate(sym.ibz2bz):
        o, v = occ[k], vir[k]
        c = c1[i_ibz]
        c[:, o] += sums.cb[i_ibz]
        c[numpy.ix_(v, o)] += sums.y1[i_ibz]
        if second:
            c[numpy.ix_(o, o)] += sums.coo[i_ibz]
            c[o, :] -= sums.y4[i_ibz].dot(fock[k][o, :])
            c[:, v] -= fock[k][:, v].dot(y5[i_ibz][numpy.ix_(v, v)])

    if second:
        _singles_term(c1, ints, sym, sums.y1, e_o, e_v, occ, vir, coef[0], shift, exchange)
        rdm = (sums.y4, [y5[i][numpy.ix_(vir[k], vir[k])] for i, k in enumerate(sym.ibz2bz)])
        c0 = c0[0] + c0[1]
    else:
        c1 = [c * .5 for c in c1]
        c0 = c0[0] * .5
        rdm = None
    v = [c + c.conj().T for c in c1]
    return v, float(c0), rdm


def _singles_term(c1, ints, sym, y1_ibz, e_o, e_v, occ, vir, coef, shift, exchange):
    '''sum_{ki} sum_{ia} y1[a,i](ki) conj(tbar[i,a,j,b])(ki,ki,kj,kj) -> [j,b] on kj.

    y1 = 2 f_ia tbar is itself a sum over all quadruples, so that this term is
    evaluated in a second, cheap pass over the momentum-diagonal amplitudes.
    '''
    nk = sym.nkpts
    a1, b1 = coef
    y1 = sym.expand_matrix(y1_ibz)
    if max(abs(y).max() if y.size else 0. for y in y1_ibz) < 1e-14:
        return
    my_ki = mpi.split(range(nk))
    out = []
    for i_ibz, kj in enumerate(sym.ibz2bz):
        row_j = ints.row(kj)
        acc = numpy.zeros((len(occ[kj]), len(vir[kj])), dtype=numpy.complex128)
        for ki in my_ki:
            row_i = ints.row(ki)
            g = _ovov(row_i[ki], row_j[kj], 1. / nk)
            if g.size == 0:
                continue
            e = _denominator(e_o[ki][:, None] - e_v[ki], e_o[kj][:, None] - e_v[kj], shift)
            tb = a1 * (g / e)
            if exchange:
                gx = _ovov(row_i[kj], row_j[ki], 1. / nk)
                tb -= b1 * (gx.transpose(0, 3, 2, 1) / e)
            no_i, nv_a, no_j, nv_b = tb.shape
            # tb is conj(tbar) here
            acc += y1[ki].T.ravel().dot(tb.reshape(no_i * nv_a, no_j * nv_b)).reshape(no_j, nv_b)
        out.append(acc)
    out = mpi.allreduce(out)
    for i_ibz, k in enumerate(sym.ibz2bz):
        c1[i_ibz][numpy.ix_(occ[k], vir[k])] += out[i_ibz]
