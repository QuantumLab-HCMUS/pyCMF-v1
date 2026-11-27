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
Orbital-optimized double hybrids (OO-DH), unrestricted.

The energy of a double hybrid,

    E[C] = E_KS[a_x HF + (1 - a_x) X, (1 - a_c) C; D]  +  a_c E_2[C],

is minimized with respect to the orbitals.  E_2 is the MP2 energy built
with the Kohn-Sham Fock matrix of the same orbitals, in semicanonical form
(the occupied-occupied and virtual-virtual blocks of the Fock matrix are
diagonalized; the occupied-virtual block, i.e. the singles, is left out).
This is the Lagrangian of OO-MP2 with a Kohn-Sham reference:
U. Bozkaya, J. Chem. Phys. 135, 224103 (2011); R. Peverati and
M. Head-Gordon, J. Chem. Phys. 139, 024110 (2013) [OO-DH];
F. Neese et al., J. Chem. Theory Comput. 5, 3060 (2009) [OO-RI-B2PLYP].

The energy is variational in the orbitals.  With the Hylleraas functional
written in spin orbitals,

    E_2 = 1/2 sum t_ijab <ij||ab> + sum_pq f_pq gamma_pq,

the derivative with respect to the rotation x_ai (phi_i -> phi_i + x_ai
phi_a) is

    dE/dx_ai = 2 f_ai + a_c [ sum_jbc t_ijbc <aj||bc> - sum_jkb t_jkab <jk||ib>
                              + 2 (f gamma - gamma f)_ai + 2 R[gamma]_ai ],

where gamma is the unrelaxed MP2 density correction and R[gamma] the
response of the Kohn-Sham Fock matrix to it (Coulomb, scaled exchange and
the exchange-correlation kernel).  The last term is what makes the
gradient exact for a functional of the density.  The self-consistent loop
of pyCMF diagonalizes

    fbar = f_KS + a_c v,     v_ai = 1/2 (dE/dx_ai - 2 f_ai) / a_c,  v_ij = v_ab = 0,

so that the converged fbar_ai is half the orbital gradient.

The functional is chosen as for OBDH (B2PLYP by default).  ``ax = ac = 1``
with ``xc_x = xc_c = ''`` is OO-MP2; ``ac = 0`` is hybrid DFT.  ``c_os`` and
``c_ss`` scale the two spin components (spin-component-scaled OO-DH).
Integrals are exact (four-index blocks in memory) or density fitted.
'''

import numpy
import scipy.linalg
from pyscf import ao2mo, dft, gto, lib, scf
from pyscf.lib import logger

from pycmf.core import status
from pycmf.mol.obdh import _OBDHMixin, double_hybrid
from pycmf.mol.obmp2 import Intermediates, UOBMP2


class _Blocks:
    '''Two-electron integrals (pq|rs) with p, q, r, s in the occupied (o) or
    virtual (v) space of a spin.'''

    def __init__(self, mp, orbs):
        self.orbs = orbs        # [(Co, Cv) per spin]
        self.mol = mp.mol
        self.with_df = mp.with_df
        self._b = None
        if self.with_df is not None:
            self._b = self._df_factors()

    def _df_factors(self):
        nao = self.mol.nao
        out = [{} for _ in self.orbs]
        chunks = {s: {k: [] for k in ('oo', 'ov', 'vv')} for s in range(len(self.orbs))}
        for lpq in self.with_df.loop():
            lpq = lib.unpack_tril(lpq).reshape(-1, nao, nao)
            for s, (co, cv) in enumerate(self.orbs):
                half_o = lib.einsum('Lpq,qi->Lpi', lpq, co)
                half_v = lib.einsum('Lpq,qa->Lpa', lpq, cv)
                chunks[s]['oo'].append(lib.einsum('pi,Lpj->Lij', co, half_o))
                chunks[s]['ov'].append(lib.einsum('pi,Lpa->Lia', co, half_v))
                chunks[s]['vv'].append(lib.einsum('pa,Lpb->Lab', cv, half_v))
        for s in range(len(self.orbs)):
            for k, v in chunks[s].items():
                out[s][k] = numpy.concatenate(v, axis=0)
        return out

    def get(self, s, pq, t, rs):
        '''(pq|rs), pq of spin s and rs of spin t, e.g. get(0, 'vv', 1, 'ov').'''
        if self._b is not None:
            return lib.einsum('Lpq,Lrs->pqrs', self._b[s][pq], self._b[t][rs])
        sel = {'o': 0, 'v': 1}
        cs = [self.orbs[s][sel[pq[0]]], self.orbs[s][sel[pq[1]]],
              self.orbs[t][sel[rs[0]]], self.orbs[t][sel[rs[1]]]]
        eri = ao2mo.general(self.mol, cs, compact=False)
        return eri.reshape([c.shape[1] for c in cs])


def _semicanonical(fock_mo, n):
    '''Rotation that diagonalizes the oo and vv blocks.'''
    eo, uo = scipy.linalg.eigh(fock_mo[:n, :n])
    ev, uv = scipy.linalg.eigh(fock_mo[n:, n:])
    return numpy.hstack((eo, ev)), scipy.linalg.block_diag(uo, uv)


def mp2_lagrangian(ints, mo_energy, nocc, c_os=1., c_ss=1.):
    '''MP2 energy (semicanonical orbitals), the density correction gamma and
    the integral part of the orbital gradient.

    Returns:
        e_ss, e_os : same- and opposite-spin energies (unscaled)
        gamma : [(gamma_oo, gamma_vv)] per spin, scaled
        w : [w_vo] per spin, sum t<aj||bc> - sum t<jk||ib>, scaled
    '''
    eo = [e[:n] for e, n in zip(mo_energy, nocc)]
    ev = [e[n:] for e, n in zip(mo_energy, nocc)]

    def denom(x, y):
        return (eo[x][:, None, None, None] + eo[y][None, :, None, None]
                - ev[x][None, None, :, None] - ev[y][None, None, None, :])

    t, e = {}, {}
    for x, y in ((0, 0), (0, 1), (1, 1)):
        g = ints.get(x, 'ov', y, 'ov').transpose(0, 2, 1, 3)      # (ia|jb) as [i,j,a,b]
        if x == y:
            g = g - g.transpose(0, 1, 3, 2)
            t[x, y] = g / denom(x, y)
            e[x, y] = .25 * numpy.einsum('ijab,ijab->', t[x, y], g)
        else:
            t[x, y] = g / denom(x, y)
            e[x, y] = numpy.einsum('ijab,ijab->', t[x, y], g)
    t[1, 0] = t[0, 1].transpose(1, 0, 3, 2)

    gamma, w = [], []
    for s in (0, 1):
        o = 1 - s
        tss, tos = t[s, s], t[s, o]          # tos[i_s, j_o, a_s, b_o]
        goo = -(.5 * c_ss * lib.einsum('ikab,jkab->ij', tss, tss)
                + c_os * lib.einsum('ikab,jkab->ij', tos, tos))
        gvv = (.5 * c_ss * lib.einsum('ijac,ijbc->ab', tss, tss)
               + c_os * lib.einsum('ijac,ijbc->ab', tos, tos))
        gamma.append((goo, gvv))
        # sum_jab t_kjab <cj||ab>  =  2 sum t_kjab (ca|jb) over both spins of j
        w1 = 2 * (c_ss * lib.einsum('kjab,cajb->ck', tss, ints.get(s, 'vv', s, 'ov'))
                  + c_os * lib.einsum('kjab,cajb->ck', tos, ints.get(s, 'vv', o, 'ov')))
        # sum_ijb t_ijcb <ij||kb>  =  2 sum t_ijcb (ik|jb)
        w2 = 2 * (c_ss * lib.einsum('ijcb,ikjb->ck', tss, ints.get(s, 'oo', s, 'ov'))
                  + c_os * lib.einsum('ijcb,ikjb->ck', tos, ints.get(s, 'oo', o, 'ov')))
        w.append(w1 - w2)
    return e[0, 0] + e[1, 1], e[0, 1], gamma, w


class UOODH(_OBDHMixin, UOBMP2):
    '''Orbital-optimized double hybrid with an unrestricted reference.

    Attributes are those of OBDH (``xc``, ``ax``, ``ac``, ``xc_x``, ``xc_c``,
    ``grids_level``) and of the self-consistent loop (``conv_tol``,
    ``conv_tol_grad``, ``max_cycle``, ``diis``, ``occupation``).  ``c_os`` and
    ``c_ss`` scale the opposite- and same-spin parts of E_2.

    Saved results:
        e_tot : orbital-optimized double-hybrid energy
        e_ref : Kohn-Sham part of the energy (E_SCF)
        e_pt2 : unscaled second-order energy E_2 (e_tot = e_ref + ac e_pt2)
        e_dh : non-optimized double-hybrid energy on the starting orbitals
    '''

    e_pt2 = None
    e_dh = None
    _keys = UOBMP2._keys | _OBDHMixin._obdh_keys | {'e_pt2', 'e_dh'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        status.experimental('OO-DH')
        UOBMP2.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        xc, ac = self._functional()
        log.info('XC functional = %s', xc)
        log.info('weight of the second-order energy a_c = %g', ac)
        log.info('c_os = %g  c_ss = %g', self.c_os, self.c_ss)
        log.info('occupation = %s', self.occupation)
        log.info('conv_tol = %g  conv_tol_grad = %s', self.conv_tol, self.conv_tol_grad)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('DIIS = %s, space = %d', self.diis, self.diis_space)
        log.info('integrals = %s', 'density fitting' if self.with_df is not None else 'exact')
        return self

    def check_sanity(self):
        if self.level_shift:
            raise NotImplementedError('level_shift is not defined for OO-DH: the '
                                      'energy would no longer be the double-hybrid one.')
        return super().check_sanity()

    def ks_terms(self, mo_coeff, nocc):
        '''Kohn-Sham density, Fock matrix (AO) and energy of the determinant.'''
        ks = self.get_ks()
        dm = numpy.array(self._make_dm(mo_coeff, nocc))
        h1e = ks.get_hcore()
        vxc = ks.get_veff(self.mol, dm)
        e_ks = ks.energy_elec(dm, h1e, vxc)[0] + ks.energy_nuc()
        return dm, h1e + vxc, e_ks

    def build(self, mo_energy, mo_coeff, nocc):
        '''Energy and orbital gradient for a set of orbitals (``mo_energy`` is
        not used: the amplitudes take the semicanonical Kohn-Sham energies).'''
        if self.level_shift:
            raise NotImplementedError('level_shift is not defined for OO-DH.')
        ks = self.get_ks()
        ac = self._functional()[1]
        s1e = ks.get_ovlp()
        dm, fock_ks, e_ks = self.ks_terms(mo_coeff, nocc)

        fock_mo, rot, e_sc, c_sc = [], [], [], []
        for c, f, n in zip(mo_coeff, fock_ks, nocc):
            fm = c.T.dot(f).dot(c)
            e, u = _semicanonical(fm, n)
            fock_mo.append(fm)
            rot.append(u)
            e_sc.append(e)
            c_sc.append(c.dot(u))

        ints = _Blocks(self, [(c[:, :n], c[:, n:]) for c, n in zip(c_sc, nocc)])
        e_ss, e_os, gamma, w = mp2_lagrangian(ints, e_sc, nocc, self.c_os, self.c_ss)
        e_pt2 = self.c_ss * e_ss + self.c_os * e_os

        # response of the Kohn-Sham Fock matrix to gamma
        gam_ao = numpy.array([c[:, :n].dot(go).dot(c[:, :n].T) + c[:, n:].dot(gv).dot(c[:, n:].T)
                              for c, n, (go, gv) in zip(c_sc, nocc, gamma)])
        mo_occ = [numpy.where(numpy.arange(c.shape[1]) < n, 1., 0.) for c, n in zip(c_sc, nocc)]
        vresp = ks.gen_response(mo_coeff=numpy.array(c_sc), mo_occ=numpy.array(mo_occ), hermi=1)
        r_ao = vresp(gam_ao)

        v, rdm_o, rdm_v = [], [], []
        for s, (c, u, n) in enumerate(zip(c_sc, rot, nocc)):
            f_sc = u.T.dot(fock_mo[s]).dot(u)
            f_vo = f_sc[n:, :n]
            go, gv = gamma[s]
            grad = w[s] + 2 * (f_vo.dot(go) - gv.dot(f_vo)) + 2 * c[:, n:].T.dot(r_ao[s]).dot(c[:, :n])
            v_sc = numpy.zeros_like(f_sc)
            v_sc[n:, :n] = .5 * grad
            v_sc[:n, n:] = .5 * grad.T
            v.append(u.dot(v_sc).dot(u.T))
            uo, uv = u[:n, :n], u[n:, n:]
            # relaxed second-order density correction in the basis of mo_coeff,
            # in the convention of properties.rdm (occupied block subtracted)
            rdm_o.append(-ac * uo.dot(go).dot(uo.T))
            rdm_v.append(ac * uv.dot(gv).dot(uv.T))

        im = Intermediates()
        im.e_ref = e_ks
        im.e_pt2 = e_pt2
        im.e_corr = ac * e_pt2
        im.e_tot = e_ks + ac * e_pt2
        im.e_hf = None
        im.dm_chan = list(dm)
        im.fock_ao, im.fock_mo = [], []
        for s, c in enumerate(mo_coeff):
            sc = s1e.dot(c)
            im.fock_ao.append(fock_ks[s] + ac * sc.dot(v[s]).dot(sc.T))
            im.fock_mo.append(fock_mo[s] + ac * v[s])
        im.fock_hf = None
        im.v = v
        im.c0 = None
        im.rdm = (rdm_o, rdm_v)
        im.blocks = None
        im.nocc = list(nocc)
        return im

    def kernel(self):
        mo_energy, mo_coeff, nocc = self._init_orbitals()
        im0 = self.build(mo_energy, mo_coeff, nocc)
        self.e_dh = im0.e_tot
        logger.info(self, 'non-optimized double hybrid on the starting orbitals: E = %.12g',
                    self.e_dh)
        e = UOBMP2.kernel(self)
        self.e_pt2 = self._last.e_pt2
        self._last.blocks = None
        return e

    def get_intermediates(self, with_amplitudes=False):
        if with_amplitudes:
            raise NotImplementedError('OO-DH does not store OBMP2 amplitude blocks.')
        return UOBMP2.get_intermediates(self)

    def make_rdm1(self, ao_repr=True, correlated=False):
        '''Density of the determinant, or with ``correlated=True`` the relaxed
        double-hybrid density D + a_c gamma (exact first derivative of the
        energy with respect to a one-electron perturbation, since the
        orbitals are optimized).'''
        return UOBMP2.make_rdm1(self, ao_repr, correlated)

    def spin_square(self, correlated=False):
        if correlated:
            raise NotImplementedError('correlated <S^2> is not available for OO-DH.')
        return UOBMP2.spin_square(self, False)

    def hfcc(self, atoms=None, correlated=False):
        if correlated:
            raise NotImplementedError('correlated hyperfine couplings are not available for OO-DH.')
        return UOBMP2.hfcc(self, atoms, False)


def OODH(mf, ax=None, ac=None, mo_coeff=None, mo_occ=None, xc=None, **kwargs):
    '''Orbital-optimized double hybrid.

    ``mf`` (HF or KS, restricted or unrestricted) gives the starting
    orbitals; the calculation is always unrestricted.  If ``mf`` is a
    :class:`pyscf.gto.Mole`, the Kohn-Sham equations of the hybrid part of
    the functional are solved first, so that ``e_dh`` is the usual
    (non-optimized) double-hybrid energy.  The functional is chosen as for
    :func:`pycmf.OBDH` (B2PLYP by default).
    '''
    mol = mf if isinstance(mf, gto.MoleBase) else None
    if mol is not None:
        mf = dft.UKS(mol)
    elif not isinstance(mf, scf.uhf.UHF):
        mf = mf.to_uhf()
    mp = UOODH(mf, None, mo_coeff, mo_occ)
    if xc is not None:
        if double_hybrid(xc) is not None:
            mp.set_functional(xc)
        else:
            if ax is not None:
                raise ValueError('ax has no meaning together with a functional string; '
                                 'put the fraction of HF exchange in the string.')
            mp.xc = xc
    if ax is not None:
        mp.ax = ax
    if ac is not None:
        mp.ac = ac
    mp.set(**kwargs)
    if mol is not None:
        ks = mp.get_ks()
        ks.conv_tol = min(ks.conv_tol, mp.conv_tol * .1)
        ks.kernel()
        if not ks.converged:
            logger.warn(mp, 'Kohn-Sham calculation of the hybrid part not converged.')
        mp._scf = ks
        mp.mo_coeff, mp.mo_occ = ks.mo_coeff, ks.mo_occ
        mp.e_hf = None
    return mp
