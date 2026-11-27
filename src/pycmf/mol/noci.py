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
Non-orthogonal configuration interaction (NOCI) in a basis of separately
optimized OBMP2 / O2BMP2 determinants.  EXPERIMENTAL.

The ground state and state-specific excited determinants (one per excitation
channel and spin, see :func:`make_basis`) are combined by solving

    H C = S C E

in the space they span, after removing near-linear dependencies of S
(canonical orthogonalization).  S, the bare Hamiltonian, the dipole operator
and S^2 between two determinants follow from the generalized Slater-Condon
rules in the corresponding-orbital form (cofactor products of the singular
values, no division by the overlap; Y. Lemke, J. Kussmann and C. Ochsenfeld,
J. Chem. Theory Comput. 21, 10193 (2025), Sec. 2.2-2.3 and App. A).

Three Hamiltonians are available (``hamiltonian``):

``'bare'``
    <I|H|J> with the physical Hamiltonian.  The diagonal is the HF energy of
    each OBMP2 determinant, not its OBMP2 energy.
``'overlap_symmetric'``
    H_IJ = H^bare_IJ + (D_I + D_J) S_IJ / 2 with D_I = E_I - H^bare_II, so that
    the diagonal is the OBMP2 energy.
``'state_specific_fock'`` (default)
    Each determinant carries its converged correlated Fock operator F_I and a
    constant c_I with c_I + sum_occ F_I = E_I.  The coupling uses the average
    operator, H_IJ = (c_I + c_J)/2 S_IJ + sum_s tr[(F_I,s + F_J,s)/2 D_IJ,s],
    with D_IJ,s the spin transition density.

The last two are model choices (the averaging of two state-specific operators
is an approximation); only the diagonal of the effective matrices is fixed by
the OBMP2 theory (N. T. Tran and L. N. Tran, J. Chem. Phys. 162, 104109 (2025)).
'''

import re

import numpy
import scipy.linalg
from pyscf import lib, scf
from pyscf.lib import logger

from pycmf.core import status
from pycmf.mol import excited
from pycmf.properties import transition


class Determinant:
    '''One NOCI basis determinant.

    Attributes:
        label : str
        occ_coeff : (alpha, beta) occupied orbitals in the AO basis
        e_tot : energy of the state-specific calculation (OBMP2 energy)
        fock_ao : (alpha, beta) correlated Fock operators in the AO basis, or None
        scalar : constant c with c + sum_occ <i|F|i> = e_tot, or None
        converged : convergence flag of the state-specific calculation
    '''

    def __init__(self, occ_coeff, e_tot, label='', fock_ao=None, converged=True):
        self.label = label
        self.occ_coeff = (numpy.asarray(occ_coeff[0]), numpy.asarray(occ_coeff[1]))
        self.e_tot = float(e_tot)
        self.converged = bool(converged)
        self.fock_ao = None
        self.scalar = None
        if fock_ao is not None:
            fock = [numpy.asarray(f) for f in fock_ao]
            for f in fock:
                if not numpy.allclose(f, f.T, atol=1e-8):
                    raise ValueError('The Fock operator of %s is not symmetric.' % label)
            self.fock_ao = (fock[0], fock[1])
            self.scalar = self.e_tot - sum(numpy.einsum('pi,pq,qi->', c, f, c)
                                           for c, f in zip(self.occ_coeff, self.fock_ao))

    @property
    def nelec(self):
        return self.occ_coeff[0].shape[1], self.occ_coeff[1].shape[1]

    @classmethod
    def from_state(cls, state, label=''):
        '''From a converged pyCMF object (OBMP2, O2BMP2, OBDH, ...) or a
        PySCF mean-field object (then the Fock operator is the HF one).'''
        occ = transition.occupied_orbitals(state)
        if hasattr(state, '_channels'):
            fock = state.get_intermediates().fock_ao
            if len(fock) == 1:
                fock = [fock[0], fock[0]]
            return cls(occ, state.e_tot, label, fock, state.converged)
        if isinstance(state, scf.hf.SCF):
            fock = numpy.asarray(state.get_fock())
            if fock.ndim == 2:
                fock = [fock, fock]
            return cls(occ, state.e_tot, label, fock, state.converged)
        raise TypeError('Cannot build a determinant from %r' % (state,))


def make_basis(mf_gs, channels, method=None, spins=(0, 1), skip_unconverged=True, **kwargs):
    '''Ground state and state-specific excited states for NOCI.

    Args:
        mf_gs : converged closed-shell HF (RHF or UHF) of the ground state
        channels : list of (hole, particle) or (hole, particle, gap), e.g. from
            :func:`pycmf.mol.excited.select_channels`; indices of ``mf_gs``
        method : callable ``method(mf, **kwargs)`` returning an unconverged
            pyCMF object, e.g. ``functools.partial(pycmf.O2BMP2, c_os=1.2)``;
            default ``pycmf.OBMP2``
        spins : spins to excite for every channel (default both)
        skip_unconverged : drop excited states whose correlated calculation
            did not converge (a warning is printed).  A MOM-UHF starting point
            that did not converge only gives a warning: it is a starting guess.
            States whose calculation raises a numerical error (e.g. a
            divergence to NaN) are always dropped.  The dropped labels and
            reasons are kept in ``make_basis.failed`` after the call.
        kwargs : attributes of the correlated objects.  The defaults here are
            tighter than those of the methods: ``conv_tol=1e-10``,
            ``conv_tol_grad=1e-7``, ``max_cycle=1000``.  The diagonal of the
            NOCI matrices converges quadratically with the orbital residual,
            the off-diagonal couplings only linearly.

    Returns:
        labels, states : 'GS' and 'h<hole>_p<particle>_<a|b>' labels and the
        converged objects, the ground state first
    '''
    if method is None:
        from pycmf.mol.obmp2 import OBMP2 as method
    kwargs = dict(dict(conv_tol=1e-10, conv_tol_grad=1e-7, max_cycle=1000), **kwargs)
    log = logger.new_logger(mf_gs)
    gs = method(mf_gs, **kwargs).run()
    if not gs.converged:
        raise RuntimeError('The ground state did not converge.')
    labels, states = ['GS'], [gs]
    failed = make_basis.failed = []
    for ch in channels:
        hole, particle = ch[0], ch[1]
        for spin in spins:
            label = excited.channel_label(hole, particle, spin)
            try:
                mf_es = excited.mom_excitation(mf_gs, hole, particle, spin)
                es = method(mf_es, occupation='imom', **kwargs).run()
            except (ValueError, numpy.linalg.LinAlgError, FloatingPointError) as err:
                log.warn('%s: failed (%s); skipped', label, err)
                failed.append((label, str(err)))
                continue
            if not mf_es.converged:
                log.warn('%s: MOM-UHF starting point not converged', label)
            if not (es.converged and numpy.isfinite(es.e_tot)):
                failed.append((label, 'not converged'))
                log.warn('%s: correlated calculation not converged%s', label,
                         '; skipped' if skip_unconverged else '')
                if skip_unconverged or not numpy.isfinite(es.e_tot):
                    continue
            labels.append(label)
            states.append(es)
    return labels, states


class NOCI(lib.StreamObject):
    '''NOCI in a basis of determinants.

    Args:
        states : converged pyCMF objects, PySCF mean-field objects or
            :class:`Determinant`; the first one is the reference (ground state)
        labels : names of the determinants; channel analysis uses labels of
            the form 'h<hole>_p<particle>_<a|b>' (see :func:`make_basis`)
        mol : the molecule; needed only if all states are :class:`Determinant`

    Attributes:
        hamiltonian : 'state_specific_fock' (default), 'overlap_symmetric' or 'bare'
        lindep : eigenvalues of S below this are removed.  Default 1e-7.
        direct : evaluate the two-electron terms with direct J/K builds instead
            of the AO integrals in memory.  Default: only if the integrals
            need more than half of ``max_memory``.

    Saved results (after ``kernel``):
        e : NOCI energies (Hartree), ascending
        ci : coefficients, columns normalized as C^T S C = 1
        e_exc : e - e[0]
        dipole : (3, nroots) transition dipoles from root 0 (root 0 set to 0)
        f : oscillator strengths from root 0
        spin_square : <S^2> of every root
        nlindep : number of removed directions of S
    '''

    hamiltonian = 'state_specific_fock'
    lindep = 1e-7
    direct = None

    _keys = {'hamiltonian', 'lindep', 'direct', 'mol', 'max_memory', 'determinants',
             's', 'h_bare', 'dip', 's2', 'e', 'ci', 'e_exc', 'dipole', 'f',
             'spin_square', 'nlindep', 'model'}

    def __init__(self, states, labels=None, hamiltonian=None, mol=None):
        status.experimental('NOCI')
        if len(states) < 1:
            raise ValueError('NOCI needs at least one determinant.')
        if labels is None:
            labels = [getattr(st, 'label', '') or 'D%d' % i for i, st in enumerate(states)]
        if len(labels) != len(states):
            raise ValueError('labels and states have different lengths.')
        dets = []
        for st, label in zip(states, labels):
            if isinstance(st, Determinant):
                st.label = label
                dets.append(st)
            else:
                dets.append(Determinant.from_state(st, label))
        if len({d.nelec for d in dets}) != 1:
            raise ValueError('All determinants must have the same numbers of alpha and '
                             'beta electrons: %s' % [d.nelec for d in dets])
        if mol is None:
            mol = transition._mol(*states)
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = mol.max_memory
        if hamiltonian is not None:
            self.hamiltonian = hamiltonian
        self.determinants = dets
        self.s = self.h_bare = self.dip = self.s2 = None
        self.e = self.ci = self.e_exc = self.dipole = self.f = None
        self.spin_square = None
        self.nlindep = None
        self.model = None

    @property
    def labels(self):
        return [d.label for d in self.determinants]

    # ---- matrices of the physical operators
    def _two_electron(self):
        mol = self.mol
        nao = mol.nao_nr()
        direct = self.direct
        if direct is None:
            direct = 2 * 8 * nao**4 / 1e6 > .5 * self.max_memory
        if direct:
            return _DirectJK(mol)
        return _IncoreERI(mol)

    def build(self):
        '''S, H (bare), dipole and S^2 matrices of the determinant basis.'''
        log = logger.new_logger(self)
        cput0 = (logger.process_clock(), logger.perf_counter())
        mol = self.mol
        dets = self.determinants
        n = len(dets)
        s1e = mol.intor_symmetric('int1e_ovlp')
        hcore = scf.hf.get_hcore(mol)
        with mol.with_common_orig((0, 0, 0)):
            r = mol.intor_symmetric('int1e_r', comp=3)
        e_nuc = mol.energy_nuc()
        dip_nuc = numpy.einsum('a,ax->x', mol.atom_charges(), mol.atom_coords())
        eri = self._two_electron()
        nalpha, nbeta = dets[0].nelec
        ms = (nalpha - nbeta) * .5
        for d in dets:
            for c in d.occ_coeff:
                err = abs(c.T.dot(s1e).dot(c) - numpy.eye(c.shape[1])).max(initial=0)
                if err > 1e-6:
                    raise ValueError('Occupied orbitals of %s are not orthonormal (%.1e).'
                                     % (d.label, err))

        S = numpy.empty((n, n))
        H = numpy.empty((n, n))
        D = numpy.empty((3, n, n))
        S2 = numpy.empty((n, n))
        for i in range(n):
            for j in range(i, n):
                sab, h, mu, ss = _pair(dets[i], dets[j], s1e, hcore, eri, r,
                                       e_nuc, dip_nuc, ms, nbeta)
                S[i, j] = S[j, i] = sab
                H[i, j] = H[j, i] = h
                D[:, i, j] = D[:, j, i] = mu
                S2[i, j] = S2[j, i] = ss
        if abs(numpy.diag(S) - 1).max() > 1e-6:
            raise ValueError('Determinants are not normalized.')
        self.s, self.h_bare, self.dip, self.s2 = S, H, D, S2
        log.timer('NOCI matrices (%d determinants)' % n, *cput0)
        return S, H, D, S2

    def get_hamiltonian(self, model=None):
        '''Hamiltonian matrix of the chosen model (see module docstring).'''
        if self.s is None:
            self.build()
        model = self.hamiltonian if model is None else model
        S, H_bare = self.s, self.h_bare
        e = numpy.array([d.e_tot for d in self.determinants])
        if model == 'bare':
            return H_bare.copy()
        if model == 'overlap_symmetric':
            shift = e - numpy.diag(H_bare)
            return H_bare + .5 * (shift[:, None] + shift[None, :]) * S
        if model == 'state_specific_fock':
            missing = [d.label for d in self.determinants if d.fock_ao is None]
            if missing:
                raise RuntimeError('No Fock operator for %s' % ', '.join(missing))
            s1e = self.mol.intor_symmetric('int1e_ovlp')
            n = len(self.determinants)
            H = numpy.empty((n, n))
            for i, di in enumerate(self.determinants):
                for j in range(i, n):
                    H[i, j] = H[j, i] = effective_pair(di, self.determinants[j], s1e)[1]
            err = abs(numpy.diag(H) - e).max()
            if err > 1e-8:
                raise RuntimeError('Effective diagonal differs from the state energies '
                                   'by %.2e Eh' % err)
            return H
        raise ValueError('Unknown NOCI Hamiltonian %r' % (model,))

    def kernel(self, model=None):
        '''Solve H C = S C E; returns the NOCI energies.'''
        log = logger.new_logger(self)
        model = self.hamiltonian if model is None else model
        H = self.get_hamiltonian(model)
        e, ci, nlindep = solve(H, self.s, self.lindep)
        self.model = model
        self.e, self.ci, self.nlindep = e, ci, nlindep
        self.e_exc = e - e[0]
        dipole = numpy.einsum('i,xij,jk->xk', ci[:, 0], self.dip, ci)
        dipole[:, 0] = 0.
        self.dipole = dipole
        self.f = 2. / 3 * self.e_exc * numpy.einsum('xk,xk->k', dipole, dipole)
        self.spin_square = numpy.einsum('ik,ij,jk->k', ci, self.s2, ci)
        log.note('NOCI (%s): %d determinants, %d roots, %d directions of S removed',
                 model, len(self.determinants), len(e), nlindep)
        if self.verbose >= logger.NOTE:
            self.analyze()
        return e

    def reference_weight(self):
        '''|<Phi_0|root k>|^2 for the first (reference) determinant.'''
        return (self.s[0].dot(self.ci))**2

    def channel_weights(self):
        '''Projection of every root on the alpha/beta pair of each channel.

        For a group g of determinants, W_g = b^T (S_gg)^+ b with b = (S C_k)_g.
        The groups are not orthogonal, so the weights do not add up to 1.

        Returns:
            dict {(hole, particle): array over roots}
        '''
        groups = {}
        for i, label in enumerate(self.labels):
            m = re.fullmatch(r'h([0-9]+)_p([0-9]+)_([ab])', label)
            if m:
                groups.setdefault((int(m.group(1)), int(m.group(2))), []).append(i)
        sc = self.s.dot(self.ci)
        out = {}
        for key, rows in groups.items():
            w, v = scipy.linalg.eigh(self.s[numpy.ix_(rows, rows)])
            keep = w > self.lindep
            x = (v[:, keep] / numpy.sqrt(w[keep])).T.dot(sc[rows])
            out[key] = numpy.einsum('ik,ik->k', x, x)
        return out

    def analyze(self, verbose=None):
        '''Print energies, transition moments, <S^2> and the leading channels.'''
        from pyscf.data import nist
        log = logger.new_logger(self, verbose)
        weights = self.channel_weights()
        ref = self.reference_weight()
        log.note('%4s %17s %10s %10s %10s %10s %10s %8s %8s  %s', 'root', 'E/Eh', 'dE/eV',
                 'mu_x', 'mu_y', 'mu_z', 'f', '<S^2>', '|<0|k>|^2', 'main channel (weight)')
        for k in range(len(self.e)):
            main = '-'
            if weights:
                key = max(weights, key=lambda g: weights[g][k])
                if weights[key][k] > .05:
                    main = '%d->%d (%.3f)' % (key[0], key[1], weights[key][k])
            mu = self.dipole[:, k]
            log.note('%4d %17.9f %10.5f %10.6f %10.6f %10.6f %10.6f %8.4f %8.4f  %s', k,
                     self.e[k], self.e_exc[k] * nist.HARTREE2EV, mu[0], mu[1], mu[2],
                     self.f[k], self.spin_square[k], ref[k], main)
        return self


def solve(H, S, lindep=1e-7):
    '''Generalized eigenproblem H C = S C E by canonical orthogonalization.

    Returns:
        e, C (with C^T S C = 1), number of removed directions
    '''
    w, v = scipy.linalg.eigh(S)
    if w[0] < -lindep:
        raise ValueError('S has a negative eigenvalue %.3e' % w[0])
    keep = w > lindep
    if not keep.any():
        raise ValueError('No linearly independent determinant left.')
    x = v[:, keep] / numpy.sqrt(w[keep])
    h = x.T.dot(H).dot(x)
    e, u = scipy.linalg.eigh((h + h.T) * .5)
    c = x.dot(u)
    err = abs(c.T.dot(S).dot(c) - numpy.eye(c.shape[1])).max()
    if err > 1e-7:
        raise RuntimeError('C^T S C differs from 1 by %.1e' % err)
    return e, c, int(numpy.count_nonzero(~keep))


def effective_pair(left, right, s1e):
    '''S_IJ and H_IJ of the 'state_specific_fock' model.'''
    s, (pa, pb) = transition.transition_density(left, right, s1e)
    f = [.5 * (a + b) for a, b in zip(left.fock_ao, right.fock_ao)]
    h = .5 * (left.scalar + right.scalar) * s
    h += numpy.einsum('pq,qp->', f[0], pa) + numpy.einsum('pq,qp->', f[1], pb)
    return s, float(h)


# ---- generalized Slater-Condon rules for the physical operators
class _IncoreERI:
    '''Two-electron contractions with the AO integrals in memory.'''

    def __init__(self, mol):
        nao = mol.nao_nr()
        eri = mol.intor('int2e', aosym='s1').reshape((nao,) * 4)
        self.coul = eri.reshape(nao * nao, nao * nao)
        # (ps|rq) as a matrix over (pq),(rs)
        self.exch = numpy.ascontiguousarray(eri.transpose(0, 3, 2, 1)).reshape(nao * nao, nao * nao)

    def same_spin(self, q):
        '''J[i,j] = (Q_i|Q_j), K[i,j] = exchange, Q_i[p,q] = r_i[p] l_i[q].'''
        x = q.reshape(q.shape[0], -1)
        return x.dot(self.coul).dot(x.T), x.dot(self.exch).dot(x.T)

    def cross(self, pa, pb):
        return float(pa.ravel().dot(self.coul).dot(pb.ravel()))


class _DirectJK:
    '''The same contractions from direct J/K builds (no stored integrals).'''

    def __init__(self, mol):
        self.mol = mol

    def same_spin(self, q):
        qt = q.transpose(0, 2, 1)
        vj, vk = scf.hf.get_jk(self.mol, qt, hermi=0)
        return (numpy.einsum('ipq,jpq->ij', q, vj), numpy.einsum('ipq,jpq->ij', q, vk))

    def cross(self, pa, pb):
        vj = scf.hf.get_jk(self.mol, pb.T, hermi=0, with_k=False)[0]
        return float(numpy.einsum('pq,pq->', pa, vj))


def _spin_terms(cl, cr, s1e, eri):
    '''Overlap, transition density and same-spin repulsion of one spin.'''
    nao = s1e.shape[0]
    n = cl.shape[1]
    if n == 0:
        return 1., numpy.zeros((nao, nao)), 0.
    phase, sigma, l, r = transition.corresponding_orbitals(cl, cr, s1e)
    w1 = phase * transition.cofactor_weights(sigma, 1)
    p = (r * w1).dot(l.T)
    e2 = 0.
    if n > 1:
        w2 = phase * transition.cofactor_weights(sigma, 2)
        q = numpy.einsum('pi,qi->ipq', r, l)
        j, k = eri.same_spin(q)
        e2 = .5 * float(numpy.sum(w2 * (j - k)))
    return phase * float(numpy.prod(sigma)), p, e2


def _pair(left, right, s1e, hcore, eri, r, e_nuc, dip_nuc, ms, nbeta):
    '''<L|R>, <L|H|R>, <L|mu|R> (electrons and nuclei) and <L|S^2|R>.'''
    sa, pa, ea = _spin_terms(left.occ_coeff[0], right.occ_coeff[0], s1e, eri)
    sb, pb, eb = _spin_terms(left.occ_coeff[1], right.occ_coeff[1], s1e, eri)
    s = sa * sb
    dm = sb * pa + sa * pb
    h = (e_nuc * s + numpy.einsum('pq,qp->', hcore, dm)
         + sb * ea + sa * eb + eri.cross(pa, pb))
    mu = dip_nuc * s - numpy.einsum('xpq,qp->x', r, dm)
    ss = (ms * (ms + 1) + nbeta) * s - numpy.trace(pa.dot(s1e).dot(pb).dot(s1e))
    return s, float(h), mu, float(ss)
