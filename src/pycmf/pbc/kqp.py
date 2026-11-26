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
Diagonal second-order self-energy and quasiparticle energies for the bands
of k-point OBMP2.  EXPERIMENTAL.

For a band p at k-point k the second-order self-energy has two parts,

    A_p(w) = sum_{iab} W[i,a,p,b] / (w + e_i - e_a - e_b)       (2p1h)
    B_p(w) = sum_{iaj} W[i,a,j,p] / (w + e_a - e_i - e_j)       (2h1p)

with W = 2 Re[ conj(alpha g - beta g_exchange) g ], g = (ia|pb)/Nk or
(ia|jp)/Nk, and alpha = (c_ss + c_os)/2, beta = c_ss/2.  At w = e_p,
A < 0 for an occupied band (correlation lost on removal) and B > 0
(relaxation); for a virtual band A < 0 and B > 0 as well.

Relation to OBMP2.  With canonical orbitals the diagonal of the correlation
potential is  v_hh = A_h(e_h)  for an occupied band and  v_ll = B_l(e_l)  for
a virtual band: the eigenvalues of the correlated Fock matrix contain the half
of the static second-order self-energy that opens the gap.  The IPEA
correction adds the other half, B_h(e_h) and A_l(e_l), which closes it.  The
two do not double count.

Spin-component scaling.  The amplitudes enter the correlation potential
linearly (factor c) and quadratically (factor c^2), as in the Hylleraas
functional, so that the half contained in the eigenvalues carries the factors
2 c_os - c_os^2 and 2 c_ss - c_ss^2 (0.96 and 0.56 for SCS with 1.2 and 1/3;
0.91 for SOS with 1.3), whereas the IPEA formula scales the other half with
c_os and c_ss themselves.  ``scaling='hylleraas'`` evaluates the self-energy
with the factors of the potential, which treats the two halves alike.

Schemes for the quasiparticle energy of band p (reference='obmp2'; "in" is
the half contained in the eigenvalue e_p, "out" the other one):

    static    e_p + out(e_p)                                 IPEA-KOBMP2
    linear    e_p + Z out(e_p),  Z = 1 / (1 - A'(e_p) - B'(e_p))
    dynamic   root of  w = e_p + [in(w) - in(e_p)] + out(w)

and with reference='hf' (orbitals and energies of Hartree-Fock, i.e. the
conventional second-order quasiparticle energies):

    static    e_p + A(e_p) + B(e_p)
    linear    e_p + Z [A(e_p) + B(e_p)]
    dynamic   root of  w = e_p + A(w) + B(w)

The self-energy is tabulated on a frequency grid around e_p in one pass over
the integrals; the dynamic equation is solved on a spline through the grid.
'''

import numpy
from pyscf.lib import logger

from pycmf.pbc import kintegrals


def _contract(l_first, l_second):
    '''sum_Q L[Q,p,q] L[Q,r,s] -> [p,q,r,s]'''
    naux = l_first.shape[0]
    shape = l_first.shape[1:] + l_second.shape[1:]
    if 0 in shape:
        return numpy.zeros(shape, dtype=numpy.complex128)
    out = numpy.dot(l_first.reshape(naux, shape[0] * shape[1]).T,
                    l_second.reshape(naux, shape[2] * shape[3]))
    return out.reshape(shape)


def tabulate(kt, e_p, grid, kconserv, e_o, e_v, ov, l_pv, l_op, alpha, beta):
    '''Second-order self-energy of the bands p of k-point ``kt`` on a
    frequency grid, from three-index integrals.

    Args:
        e_p : energies of the bands (centres of the grids)
        grid : frequency offsets; must contain 0
        e_o, e_v : lists over the k-points of the occupied and virtual
            orbital energies of the denominators
        ov : function (ki, ka) -> L^Q_ia(ki, ka)
        l_pv : list over kb of L^Q_pb(kt, kb)
        l_op : list over kj of L^Q_jp(kj, kt)
        alpha, beta : (c_ss + c_os)/2 and c_ss/2

    Returns:
        A, B (bands x grid) and their frequency derivatives at the band
        energies
    '''
    nk = len(e_o)
    grid = numpy.asarray(grid, dtype=float)
    i0 = int(numpy.argmin(abs(grid)))
    scale = 2. / nk**2
    npb = len(e_p)
    w_grid = numpy.asarray(e_p)[:, None] + grid[None, :]                 # [p,g]
    a = numpy.zeros((npb, grid.size))
    b = numpy.zeros((npb, grid.size))
    da = numpy.zeros(npb)
    db = numpy.zeros(npb)
    for ki in range(nk):
        for ka in range(nk):
            l_ia = ov(ki, ka)
            if l_ia.shape[1] == 0 or l_ia.shape[2] == 0:
                continue
            e_ia = e_o[ki][:, None] - e_v[ka][None, :]
            # ---- 2p1h: (i a | p b)
            kb = kconserv[ki, ka, kt]
            g = _contract(l_ia, l_pv[kb])                                   # [i,a,p,b]
            if beta != 0:
                gx = _contract(ov(ki, kb), l_pv[ka]).transpose(0, 3, 2, 1)
                w = scale * ((alpha * g - beta * gx).conj() * g).real
            else:
                w = (scale * alpha) * (g.real**2 + g.imag**2)
            w = w.transpose(2, 0, 1, 3)                                    # [p,i,a,b]
            d0 = e_ia[:, :, None] - e_v[kb][None, None, :]                  # [i,a,b]
            den = w_grid[:, :, None, None, None] + d0[None, None]
            a += numpy.einsum('piab,pgiab->pg', w, 1. / den)
            da -= numpy.einsum('piab,piab->p', w, 1. / den[:, i0]**2)
            # ---- 2h1p: (i a | j p)
            kj = kconserv[ka, ki, kt]
            g = _contract(l_ia, l_op[kj])                                   # [i,a,j,p]
            if g.size == 0:
                continue
            if beta != 0:
                gx = _contract(ov(kj, ka), l_op[ki]).transpose(2, 1, 0, 3)
                w = scale * ((alpha * g - beta * gx).conj() * g).real
            else:
                w = (scale * alpha) * (g.real**2 + g.imag**2)
            w = w.transpose(3, 0, 1, 2)                                    # [p,i,a,j]
            d0 = -e_ia[:, :, None] - e_o[kj][None, None, :]                 # [i,a,j]
            den = w_grid[:, :, None, None, None] + d0[None, None]
            b += numpy.einsum('piaj,pgiaj->pg', w, 1. / den)
            db -= numpy.einsum('piaj,piaj->p', w, 1. / den[:, i0]**2)
    return a, b, da, db


def self_energy(mp, nvir=2, kpts_index=None, grid=None, mo_energy=None, nocc=None,
                scaling='linear'):
    '''Tabulate the diagonal second-order self-energy near the band energies.

    Args:
        mp : KOBMP2 object with orbitals (after ``kernel()`` or
            ``init_from_scf()``)
        nvir : number of lowest virtual bands that are treated
        nocc : number of highest occupied bands that are treated (default:
            all correlated ones).  ``nocc=1, nvir=1`` is enough for the gap
            and three times cheaper than the default for four occupied bands.
        kpts_index : k-points (indices in the full mesh); default: the
            irreducible ones, copied to the symmetry-equivalent k-points
        grid : frequency offsets (Hartree) from the band energy; default: 41
            points within +-0.4 Hartree, or within 1.4 times the gap of the
            orbital energies if that is smaller.  The dynamic equation is
            solved on the pole-free part of the grid: above -0.7 gaps for
            occupied and below +0.7 gaps for virtual band-edge states.
        mo_energy : orbital energies used in the denominators and as centres
            of the frequency grids (default: ``mp.mo_energy``)
        scaling : spin-component factors of the self-energy.  'linear': c_os
            and c_ss (the IPEA formula).  'hylleraas': 2c - c^2 for each
            component, the factors with which the correlation potential of
            OBMP2 itself carries the self-energy (see above); the two are the
            same for c_os = c_ss = 1.

    Returns:
        list over the k-points of the mesh of dicts (None if not treated) with
        'occ' (bool, per band), 'e' (band energies), 'grid', 'A', 'B'
        (bands x grid), 'A0', 'B0' (values at the band energy) and 'dA', 'dB'
        (frequency derivatives there).
    '''
    log = logger.new_logger(mp)
    cput0 = (logger.process_clock(), logger.perf_counter())
    sym = mp._sym
    nk = sym.nkpts
    kconserv = mp._kconserv()
    mo_coeff = [numpy.asarray(c) for c in mp.mo_coeff]
    if mo_energy is None:
        mo_energy = mp.mo_energy
    mo_energy = [numpy.asarray(e) for e in mo_energy]
    nelec = [int(numpy.count_nonzero(o > 0)) for o in mp.mo_occ]
    nmo = [c.shape[1] for c in mo_coeff]
    occ, vir = mp._active(nelec, nmo)
    if mp.with_df is None:
        mp.with_df = kintegrals.as_gdf(mp._scf)
    ints = kintegrals.KDFInts(mp.with_df, mp.kpts, mp.max_memory, mp.verbose, mp.stdout)
    ints.single_precision = getattr(mp, 'single_precision', False)
    ints.update(mo_coeff, occ, vir)
    if scaling == 'linear':
        c_os, c_ss = mp.c_os, mp.c_ss
    elif scaling == 'hylleraas':
        c_os, c_ss = 2 * mp.c_os - mp.c_os**2, 2 * mp.c_ss - mp.c_ss**2
    else:
        raise ValueError(scaling)
    alpha, beta = .5 * (c_ss + c_os), .5 * c_ss
    e_o = [mo_energy[k][occ[k]] for k in range(nk)]
    e_v = [mo_energy[k][vir[k]] for k in range(nk)]
    # Poles of the self-energy of a band-edge state: one gap of the orbital
    # energies away on the side the quasiparticle energy moves away from, two
    # gaps on the side it moves to (up for occupied, down for virtual bands).
    top = max(e.max() for e in e_o if e.size)
    bottom = min(e.min() for e in e_v if e.size)
    gap0 = bottom - top
    if grid is None:
        half = min(.4, 1.4 * gap0)
        grid = numpy.linspace(-half, half, 41)
    grid = numpy.asarray(grid, dtype=float)
    i0 = int(numpy.argmin(abs(grid)))
    if abs(grid[i0]) > 1e-12:
        raise ValueError('the frequency grid must contain 0')
    targets = [int(k) for k in (sym.ibz2bz if kpts_index is None else kpts_index)]
    out = [None] * nk
    for kt in targets:
        n_l = min(nvir, len(vir[kt]))
        occ_t = occ[kt] if nocc is None else occ[kt][len(occ[kt]) - min(nocc, len(occ[kt])):]
        bands = numpy.concatenate([occ_t, vir[kt][:n_l]])
        is_occ = numpy.arange(len(bands)) < len(occ_t)
        e_p = mo_energy[kt][bands]
        c_p = mo_coeff[kt][:, bands]
        l_pv = [ints.block(kt, kb, c_p, mo_coeff[kb][:, vir[kb]]) for kb in range(nk)]
        l_op = [ints.block(kj, kt, mo_coeff[kj][:, occ[kj]], c_p) for kj in range(nk)]
        a, b, da, db = tabulate(kt, e_p, grid, kconserv, e_o, e_v, ints.ov, l_pv, l_op,
                                alpha, beta)
        out[kt] = {'occ': is_occ, 'e': e_p, 'grid': grid, 'A': a, 'B': b,
                   'A0': a[:, i0].copy(), 'B0': b[:, i0].copy(), 'dA': da, 'dB': db, 'gap': gap0}
    if kpts_index is None:
        for k, i in enumerate(sym.bz2ibz):
            out[k] = out[sym.ibz2bz[i]]
    log.timer('second-order self-energy', *cput0)
    return out


def _root(grid, f, guess):
    '''Root of f (tabulated on grid) closest to guess; nan if there is none.'''
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq
    spline = CubicSpline(grid, f)
    sign = numpy.sign(f)
    idx = numpy.where(sign[:-1] * sign[1:] <= 0)[0]
    if len(idx) == 0:
        return numpy.nan
    roots = [brentq(spline, grid[i], grid[i + 1]) for i in idx]
    return roots[int(numpy.argmin([abs(r - guess) for r in roots]))]


def quasiparticle(se, scheme='static', reference='obmp2', e_ref=None):
    '''Quasiparticle energies from a tabulated self-energy.

    Args:
        se : output of :func:`self_energy`
        scheme : 'static', 'linear' or 'dynamic'
        reference : 'obmp2' (the band energies contain one half of the static
            self-energy) or 'hf' (they contain none)
        e_ref : reference band energies (list over k-points, same layout as
            ``se[k]['e']``) if they differ from the energies the self-energy
            was built with

    Returns:
        e_qp, z : lists over the k-points of arrays over the bands
    '''
    if scheme not in ('static', 'linear', 'dynamic'):
        raise ValueError(scheme)
    if reference not in ('obmp2', 'hf'):
        raise ValueError(reference)
    e_qp, zs = [], []
    for k, s in enumerate(se):
        if s is None:
            e_qp.append(None)
            zs.append(None)
            continue
        e0 = s['e'] if e_ref is None else numpy.asarray(e_ref[k])
        occ = s['occ']
        z = 1. / (1. - s['dA'] - s['dB'])
        if reference == 'obmp2':
            inside = numpy.where(occ[:, None], s['A'], s['B'])
            outside = numpy.where(occ[:, None], s['B'], s['A'])
            in0 = numpy.where(occ, s['A0'], s['B0'])
            out0 = numpy.where(occ, s['B0'], s['A0'])
            shift = inside - in0[:, None] + outside                        # [p,g]
        else:
            out0 = s['A0'] + s['B0']
            shift = s['A'] + s['B']
        if scheme == 'static':
            e = e0 + out0
        elif scheme == 'linear':
            e = e0 + z * out0
        else:
            # w = e0 + shift(w - e_centre); the grid is centred on s['e']
            e = numpy.empty_like(e0)
            safe = .7 * s.get('gap', numpy.inf)
            for p in range(len(e0)):
                f = e0[p] + shift[p] - (s['e'][p] + s['grid'])
                # only the part of the grid without poles for this kind of band
                keep = s['grid'] >= -safe if occ[p] else s['grid'] <= safe
                d = _root(s['grid'][keep], f[keep], z[p] * out0[p] + e0[p] - s['e'][p])
                e[p] = s['e'][p] + d
        e_qp.append(e)
        zs.append(z)
    return e_qp, zs


def band_edges(se, e_qp):
    '''Quasiparticle energies of the highest occupied and of the lowest virtual
    band (in the order of the reference energies) at every k-point.

    Only these bands are used for gaps: for bands far from the gap the static
    denominators can vanish and the correction is meaningless.
    '''
    top, bottom = [], []
    for s, e in zip(se, e_qp):
        occ = s['occ']
        top.append(e[occ][numpy.argmax(s['e'][occ])] if occ.any() else -numpy.inf)
        bottom.append(e[~occ][numpy.argmin(s['e'][~occ])] if (~occ).any() else numpy.inf)
    return numpy.array(top), numpy.array(bottom)


def gaps(se, e_qp):
    '''(direct gap at the first k-point, smallest gap on the mesh, index of
    the k-point of the valence band maximum, of the conduction band minimum)'''
    v, c = band_edges(se, e_qp)
    kv, kc = int(numpy.argmax(v)), int(numpy.argmin(c))
    return c[0] - v[0], c[kc] - v[kv], kv, kc
