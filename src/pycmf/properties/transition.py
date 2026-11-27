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
Transition properties between two separately optimized determinants.  EXPERIMENTAL.

The ground and excited states of a state-specific calculation (for example
``pycmf.OBMP2(mf_gs)`` and ``pycmf.OBMP2(mf_mom, occupation='imom')``) are two
single determinants with different, non-orthogonal orbitals.  Their transition
density follows from the generalized Slater-Condon rules: for each spin,

    X = C_L^T S C_R = U diag(sigma) V^T,      s = det(X),
    P = phase * sum_i [prod_{k != i} sigma_k] |r_i><l_i|,

with l = C_L U and r = C_R V the corresponding orbitals; P is the
cofactor-weighted transition density, valid also when X is singular.  The two
determinants are made orthogonal by the symmetric (Loewdin) transformation
before the transition dipole is taken (Bourne Worster, Feighan and Manby,
J. Chem. Phys. 154, 124106 (2021), Eqs. (3), (5)-(9)), and the oscillator
strength is f = 2/3 dE |mu|^2 in atomic units.

The result is a determinant-level transition moment.  The correlated
amplitudes of OBMP2 enter through the orbitals only, and the excited
determinant of a closed-shell molecule is a spin-mixed (S^2 ~ 1) state; see
``singlet`` in :func:`transition_dipole`.
'''

import numpy
import scipy.linalg
from pyscf import scf

from pycmf.core import status


def occupied_orbitals(state):
    '''Occupied orbitals (alpha, beta) in the AO basis.

    ``state`` is a converged pyCMF object (OBMP2, O2BMP2, OBDH, ...), a PySCF
    mean-field object, an object with ``occ_coeff`` (e.g. a NOCI determinant),
    or a pair of arrays.  For a restricted reference both spins share the
    orbitals.
    '''
    if isinstance(state, (tuple, list)) and len(state) == 2:
        return numpy.asarray(state[0]), numpy.asarray(state[1])
    occ = getattr(state, 'occ_coeff', None)
    if occ is not None:
        return occ
    if hasattr(state, '_channels'):  # pyCMF one-body correlated methods
        _, mo_coeff, nocc = state._channels()
        occ = [c[:, :n] for c, n in zip(mo_coeff, nocc)]
        if len(occ) == 1:
            occ = occ * 2
        return occ[0], occ[1]
    if isinstance(state, scf.hf.SCF):
        mo_coeff = numpy.asarray(state.mo_coeff)
        mo_occ = numpy.asarray(state.mo_occ)
        if mo_coeff.ndim == 2:  # restricted
            c = mo_coeff[:, mo_occ > 0]
            return c, c
        return mo_coeff[0][:, mo_occ[0] > 0], mo_coeff[1][:, mo_occ[1] > 0]
    raise TypeError('Cannot get occupied orbitals from %r' % (state,))


def corresponding_orbitals(cl, cr, s1e):
    '''Corresponding orbitals of two sets of occupied orbitals of one spin.

    Returns:
        phase : det(U) det(V^T), so that det(X) = phase * prod(sigma)
        sigma : singular values of X = cl^T S cr
        l, r : corresponding orbitals cl U and cr V
    '''
    x = cl.T.dot(s1e).dot(cr)
    u, sigma, vt = scipy.linalg.svd(x)
    phase = float(numpy.linalg.det(u) * numpy.linalg.det(vt))
    return phase, sigma, cl.dot(u), cr.dot(vt.T)


def cofactor_weights(sigma, skip=1):
    '''Products of the singular values with ``skip`` (1 or 2) of them left
    out: w[i] = prod_{k != i} sigma_k, or w[i, j] = prod_{k != i, j} sigma_k
    (zero on the diagonal).'''
    n = len(sigma)
    if skip == 1:
        return numpy.array([numpy.prod(numpy.delete(sigma, i)) for i in range(n)])
    w = numpy.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            w[i, j] = w[j, i] = numpy.prod(numpy.delete(sigma, (i, j)))
    return w


def spin_transition_density(cl, cr, s1e):
    '''Overlap s = <L|R> of one spin and the cofactor-weighted transition
    density P (AO), so that <L|sum o|R> = tr(o P) for a one-electron
    operator o of that spin.'''
    nao = s1e.shape[0]
    if cl.shape[1] != cr.shape[1]:
        raise ValueError('The two determinants have different numbers of electrons.')
    if cl.shape[1] == 0:
        return 1., numpy.zeros((nao, nao))
    phase, sigma, l, r = corresponding_orbitals(cl, cr, s1e)
    w = phase * cofactor_weights(sigma)
    return phase * float(numpy.prod(sigma)), (r * w).dot(l.T)


def transition_density(left, right, s1e=None):
    '''Overlap of two determinants and their spin-resolved transition density.

    Returns:
        s : <L|R>
        (pa, pb) : AO matrices with <L|sum o|R> = tr(o pa) + tr(o pb), i.e.
            each spin density already carries the overlap of the other spin
    '''
    cla, clb = occupied_orbitals(left)
    cra, crb = occupied_orbitals(right)
    if s1e is None:
        s1e = _mol(left, right).intor_symmetric('int1e_ovlp')
    sa, pa = spin_transition_density(cla, cra, s1e)
    sb, pb = spin_transition_density(clb, crb, s1e)
    return sa * sb, (sb * pa, sa * pb)


def orthogonalize(dm_l, dm_r, dm_lr, s):
    '''Transition density between the symmetrically orthogonalized states.

    With L' = c1 L + c2 R, R' = c2 L + c1 R and <L'|R'> = 0 (Loewdin),
    <L'|O|R'> = c1 c2 (O_LL + O_RR) + c1^2 O_LR + c2^2 O_RL, written as in
    Eq. (9) of Bourne Worster et al.:

        a = sqrt[(1 + s)/(1 - s)],
        D = [(1 - a^2)(D_L + D_R) + (1 + a)^2 D_LR + (1 - a)^2 D_LR^T] / [4 (1 + s)].

    Args:
        dm_l, dm_r : densities of the two determinants (AO, any spin sum)
        dm_lr : transition density <L|.|R> with the same spin sum
        s : overlap <L|R>
    '''
    if abs(s) >= 1 - 1e-8:
        raise ValueError('The two determinants are (nearly) identical, |<L|R>| = %.10f; '
                         'they cannot be orthogonalized.' % abs(s))
    a2 = (1 + s) / (1 - s)
    a = numpy.sqrt(a2)
    return ((1 - a2) * (dm_l + dm_r) + (1 + a)**2 * dm_lr
            + (1 - a)**2 * dm_lr.T) / (4 * (1 + s))


def oscillator_strength(e_exc, dipole):
    '''f = 2/3 dE |mu|^2 (atomic units; dE in Hartree, mu in e a0).'''
    return 2. / 3 * e_exc * float(numpy.dot(dipole, dipole))


class TransitionResult(dict):
    '''Result of :func:`transition_dipole`; keys are also attributes.

    e_exc : excitation energy E(es) - E(gs) in Hartree
    dipole : transition dipole <gs'|-r|es'> (e a0), times sqrt(2) if singlet
    dipole_det : the same without the singlet factor
    f : oscillator strength
    overlap : <gs|es> before orthogonalization
    spin_square : (<S^2> of gs, <S^2> of es) of the two determinants
    '''
    __getattr__ = dict.__getitem__


def transition_dipole(gs, es, singlet=False, e_gs=None, e_es=None, origin=None):
    '''Transition dipole moment and oscillator strength between two states.

    Args:
        gs, es : converged pyCMF objects (or PySCF mean-field objects); the
            energies are their ``e_tot`` unless ``e_gs``/``e_es`` are given.
        singlet : multiply the determinant moment by sqrt(2).  This
            approximates the singlet combination of the alpha and the beta
            excitation by assuming that both determinants give the same
            moment and are orthogonal to each other; it is not a spin
            projection.  Use :class:`pycmf.mol.noci.NOCI` with both branches to
            avoid the assumption.
        origin : gauge origin of the position operator (irrelevant for the
            transition moment of orthogonal states; default (0, 0, 0)).

    Returns:
        :class:`TransitionResult`
    '''
    status.experimental('transition_dipole')
    mol = _mol(gs, es)
    s1e = mol.intor_symmetric('int1e_ovlp')
    cga, cgb = occupied_orbitals(gs)
    cea, ceb = occupied_orbitals(es)
    if (cga.shape[1], cgb.shape[1]) != (cea.shape[1], ceb.shape[1]):
        raise ValueError('The two states have different numbers of alpha/beta electrons.')
    s, (pa, pb) = transition_density((cga, cgb), (cea, ceb), s1e)
    dm_gs = cga.dot(cga.T) + cgb.dot(cgb.T)
    dm_es = cea.dot(cea.T) + ceb.dot(ceb.T)
    dm = orthogonalize(dm_gs, dm_es, pa + pb, s)
    with mol.with_common_orig((0, 0, 0) if origin is None else origin):
        r = mol.intor_symmetric('int1e_r', comp=3)
    dipole_det = -numpy.einsum('xpq,qp->x', r, dm)
    dipole = dipole_det * numpy.sqrt(2) if singlet else dipole_det.copy()

    e_gs = gs.e_tot if e_gs is None else e_gs
    e_es = es.e_tot if e_es is None else e_es
    e_exc = float(e_es - e_gs)
    return TransitionResult(
        e_exc=e_exc, dipole=dipole, dipole_det=dipole_det,
        f=oscillator_strength(e_exc, dipole), overlap=float(s),
        spin_square=(float(scf.uhf.spin_square((cga, cgb), s1e)[0]),
                     float(scf.uhf.spin_square((cea, ceb), s1e)[0])))


def _mol(*states):
    for st in states:
        mol = getattr(st, 'mol', None)
        if mol is not None:
            return mol
    raise ValueError('Cannot find the molecule; pass objects with a .mol attribute.')
