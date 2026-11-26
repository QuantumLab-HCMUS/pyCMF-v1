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
Correlation potential of OBMP2.

The OBMP2 Hamiltonian is the cumulant-reduced, second-order canonical
transformation of the electronic Hamiltonian,

    H_OBMP2 = H_HF + [H, A]_1 + 1/2 [[F, A], A]_1 = F + v + const,

where A is the anti-Hermitian doubles operator built from the first-order
amplitudes and [...]_1 denotes the reduction to one-body operators and
constants.  This module evaluates the one-body potential ``v`` and the
constant for an arbitrary list of amplitude spin blocks, so that restricted
and unrestricted references share the same code.

References:
    T. N. Lan and T. Yanai, J. Chem. Phys. 138, 224108 (2013)
    L. N. Tran, J. Phys. Chem. A 125, 9242 (2021)
'''

import numpy
from pyscf import lib


def _group_tbar(blocks):
    tbar = {}
    for x, y, _, tb in blocks:
        tbar[x, y] = tbar[x, y] + tb if (x, y) in tbar else tb
    return tbar


def first_order(ints, blocks, fock, nocc, weight=1.):
    '''[H, A]_1: terms linear in the amplitudes.

    Returns:
        c0 : constant
        c1 : list of (unsymmetrised) one-body matrices, one per spin channel
    '''
    c0, c1 = ints.first_order(_group_tbar(blocks), weight)
    for x, y, _, tb in blocks:
        nx, ny = nocc[x], nocc[y]
        c1[y][:ny, ny:] += 2 * lib.einsum('iajb,ia->jb', tb, fock[x][:nx, nx:])
    return c0, c1


def second_order(blocks, fock, nocc, weight=1.):
    '''1/2 [[F, A], A]_1: terms quadratic in the amplitudes.

    Every term is a contraction of the amplitudes with the Fock matrix, so no
    two-electron integral is needed.  The cost is O(o^2 v^3) per block.
    '''
    nchan = len(fock)
    nmo = [f.shape[0] for f in fock]
    c0 = 0.
    c1 = [numpy.zeros((n, n)) for n in nmo]
    r_occ = [0] * nchan
    r_vir = [0] * nchan

    # singles-like term f_ia tbar tbar
    y1 = [0] * nchan
    for x, y, _, tb in blocks:
        y1[y] = y1[y] + lib.einsum('ia,iajb->jb', fock[x][:nocc[x], nocc[x]:], tb)
    for x, y, _, tb in blocks:
        nx = nocc[x]
        c1[x][:nx, nx:] += lib.einsum('iajb,jb->ia', tb, y1[y])

    for x, y, t, tb in blocks:
        nx, ny = nocc[x], nocc[y]
        fx = fock[x]
        fv = lib.einsum('ac,icjd->iajd', fx[nx:, nx:], tb)
        c1[y][:ny, :ny] += lib.einsum('iajb,iakb->jk', t, fv)
        c1[x][:nx, :nx] += lib.einsum('iajb,kajb->ik', t, fv)
        c0 -= weight * numpy.dot(t.ravel(), fv.ravel())
        fo = lib.einsum('ik,kajd->iajd', fx[:nx, :nx], tb)
        c1[y][:ny, :ny] -= lib.einsum('iajb,ialb->jl', t, fo)
        c1[x][nx:, nx:] += lib.einsum('iajb,icjb->ac', t, fo)
        c0 += weight * numpy.dot(t.ravel(), fo.ravel())
        fo -= fv
        fv = None
        c1[y][ny:, ny:] += lib.einsum('iajb,iajd->bd', t, fo)
        fo = None
        r_occ[x] = r_occ[x] + lib.einsum('iajb,kajb->ik', t, tb)
        r_vir[x] = r_vir[x] + lib.einsum('iajb,icjb->ac', t, tb)

    for s in range(nchan):
        if isinstance(r_occ[s], numpy.ndarray):
            ns = nocc[s]
            c1[s][:, :ns] -= fock[s][:, :ns].dot(r_occ[s].T)
            c1[s][:, ns:] -= fock[s][:, ns:].dot(r_vir[s])
    return c0, c1, (r_occ, r_vir)


def correlation_potential(ints, blocks, fock, nocc, bch_order=2, weight=1.):
    '''One-body correlation potential and constant of OBMP2.

    Args:
        ints : integral backend (see :mod:`pycmf.mol.integrals`)
        blocks : amplitude blocks (see :mod:`pycmf.core.amplitudes`)
        fock : list of HF Fock matrices in the current MO basis
        nocc : list of occupation numbers
        bch_order : 2 (default) for the full second-order expression; 1 for
            the half-weighted first commutator H + 1/2 [H, A]_1
        weight : weight of the constants (2 for a restricted reference, where
            the spin-flipped blocks are not built)

    Returns:
        v : list of symmetric matrices, one per spin channel
        c0 : constant
        rdm : (r_occ, r_vir) intermediates of the unrelaxed MP2-like density
            (None for bch_order = 1)
    '''
    nchan = len(fock)
    if not blocks:
        return [numpy.zeros_like(f) for f in fock], 0., None
    c0, c1 = first_order(ints, blocks, fock, nocc, weight)
    rdm = None
    if bch_order == 2:
        c0_2, c1_2, rdm = second_order(blocks, fock, nocc, weight)
        c0 += c0_2
        c1 = [a + b for a, b in zip(c1, c1_2)]
    elif bch_order == 1:
        c0 *= .5
        c1 = [a * .5 for a in c1]
    else:
        raise ValueError('bch_order must be 1 or 2')
    v = [(c1[s] + c1[s].T) * .5 for s in range(nchan)]
    return v, c0, rdm


def density_intermediates(blocks, nchan):
    '''Occupied-occupied and virtual-virtual parts of the unrelaxed
    second-order density, per spin channel:

        r_occ[i,k] = sum_ajb t[i,a,j,b] tbar[k,a,j,b]
        r_vir[a,c] = sum_ijb t[i,a,j,b] tbar[i,c,j,b]

    so that gamma_oo = 1 - r_occ and gamma_vv = r_vir.
    '''
    r_occ = [0] * nchan
    r_vir = [0] * nchan
    for x, y, t, tb in blocks:
        r_occ[x] = r_occ[x] + lib.einsum('iajb,kajb->ik', t, tb)
        r_vir[x] = r_vir[x] + lib.einsum('iajb,icjb->ac', t, tb)
    return r_occ, r_vir
