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
Two-electron integrals needed by OBMP2 for molecules.

Only three kinds of MO integrals enter the correlation potential,

    (ia|jb),   (ia|pb),   (ia|jp),       p = any orbital,

so the full (pq|rs) tensor is never built.  Two backends share one interface:

``IncoreInts``
    exact four-centre integrals.  The first pair is transformed once per
    spin channel and iteration, (ia|mu nu); the second pair is transformed on
    demand.
``DFInts``
    density-fitted integrals, (ia|jb) ~ sum_Q B^Q_ia B^Q_jb.

The interface consists of ``update(mo_coeff, nocc)``, ``ovov(x, y)`` and
``first_order(tbar)``, the latter returning the amplitude-linear part of the
correlation potential (see :mod:`pycmf.mol.bch`).
'''

import numpy
from pyscf import ao2mo, df, lib
from pyscf.ao2mo import _ao2mo


class _IntsBase:
    def __init__(self, mp):
        self.mol = mp.mol
        self.max_memory = mp.max_memory
        self.verbose = mp.verbose
        self.stdout = mp.stdout
        self.mo_coeff = self.nocc = self.nmo = None

    def _set_orbitals(self, mo_coeff, nocc):
        self.mo_coeff = [numpy.asarray(c, order='F') for c in mo_coeff]
        self.nocc = list(nocc)
        self.nmo = [c.shape[1] for c in mo_coeff]

    def _shape_ovov(self, x, y):
        return (self.nocc[x], self.nmo[x] - self.nocc[x],
                self.nocc[y], self.nmo[y] - self.nocc[y])


class IncoreInts(_IntsBase):
    '''Exact integrals held in memory.'''

    def __init__(self, mp):
        _IntsBase.__init__(self, mp)
        eri = getattr(mp._scf, '_eri', None)
        # number of basis functions from the orbitals, so that mean-field
        # objects with user-defined integrals (mf._eri) are supported
        nao = numpy.asarray(mp.mo_coeff).shape[-2]
        if eri is None:
            nao_pair = nao * (nao + 1) // 2
            if nao_pair**2 * 4 / 1e6 > self.max_memory:
                raise RuntimeError(
                    'The AO integrals do not fit in max_memory (%d MB).  Use density '
                    'fitting: pycmf.OBMP2(mf).density_fit().' % self.max_memory)
            eri = self.mol.intor('int2e', aosym='s8')
        self._eri = ao2mo.restore(4, eri, nao)
        self._half = None

    def update(self, mo_coeff, nocc):
        '''(ia|mu nu) for every spin channel, packed in the AO pair.'''
        self._set_orbitals(mo_coeff, nocc)
        self._half = []
        for c, n in zip(self.mo_coeff, self.nocc):
            if n == 0 or n == c.shape[1]:
                self._half.append(numpy.zeros((0, self._eri.shape[1])))
            else:
                self._half.append(
                    ao2mo.incore.half_e1(self._eri, (c[:, :n], c[:, n:]), compact=False))
        return self

    def _e2(self, x, y, i0, i1, j0, j1):
        out = _ao2mo.nr_e2(self._half[x], self.mo_coeff[y], (i0, i1, j0, j1), aosym='s2')
        return out.reshape(self.nocc[x], self.nmo[x] - self.nocc[x], i1 - i0, j1 - j0)

    def ovov(self, x, y):
        return self._e2(x, y, 0, self.nocc[y], self.nocc[y], self.nmo[y])

    def first_order(self, tbar, weight=1.):
        '''Amplitude-linear two-electron terms.

        Args:
            tbar : dict {(x, y): tbar[i,a,k,l]} of antisymmetrised amplitudes
            weight : weight of the scalar term (2 for a restricted reference)
        Returns:
            c0, [c1 per channel] with

                c1[y][p,k] += 2 (ia|pl) tbar[ia,kl]
                c1[y][p,l] -= 2 (ia|kp) tbar[ia,kl]
                c0         -=   (ia|kl) tbar[ia,kl]
        '''
        c0 = 0.
        c1 = [numpy.zeros((n, n)) for n in self.nmo]
        for (x, y), tb in tbar.items():
            no, nmo = self.nocc[y], self.nmo[y]
            ovgv = self._e2(x, y, 0, nmo, no, nmo)
            c1[y][:, :no] += 2 * lib.einsum('iapl,iakl->pk', ovgv, tb)
            c0 -= weight * numpy.dot(ovgv[:, :, :no].ravel(), tb.ravel())
            ovgv = None
            ovog = self._e2(x, y, 0, no, 0, nmo)
            c1[y][:, no:] -= 2 * lib.einsum('iakp,iakl->pl', ovog, tb)
            ovog = None
        return c0, c1


class DFInts(_IntsBase):
    '''Density-fitted integrals; three-index MO tensors are held in memory.'''

    def __init__(self, mp):
        _IntsBase.__init__(self, mp)
        self.with_df = mp.with_df
        self._bgv = self._bog = None

    def update(self, mo_coeff, nocc):
        '''B^Q_pa (general-virtual) and B^Q_ip (occupied-general).'''
        self._set_orbitals(mo_coeff, nocc)
        naux = self.with_df.get_naoaux()
        nchan = len(self.nmo)
        self._bgv = [numpy.empty((naux, n, n - o)) for n, o in zip(self.nmo, self.nocc)]
        self._bog = [numpy.empty((naux, o, n)) for n, o in zip(self.nmo, self.nocc)]
        p0 = 0
        for eri1 in self.with_df.loop():
            p1 = p0 + eri1.shape[0]
            for s in range(nchan):
                c, n, o = self.mo_coeff[s], self.nmo[s], self.nocc[s]
                if o == n:
                    continue
                self._bgv[s][p0:p1] = _ao2mo.nr_e2(
                    eri1, c, (0, n, o, n), aosym='s2').reshape(p1 - p0, n, n - o)
                if o > 0:
                    self._bog[s][p0:p1] = _ao2mo.nr_e2(
                        eri1, c, (0, o, 0, n), aosym='s2').reshape(p1 - p0, o, n)
            p0 = p1
        return self

    def _bov(self, s):
        naux = self._bgv[s].shape[0]
        return self._bgv[s][:, :self.nocc[s]].reshape(naux, -1)

    def ovov(self, x, y):
        return lib.dot(self._bov(x).T, self._bov(y)).reshape(self._shape_ovov(x, y))

    def first_order(self, tbar, weight=1.):
        c0 = 0.
        c1 = [numpy.zeros((n, n)) for n in self.nmo]
        for y in range(len(self.nmo)):
            no, nv = self.nocc[y], self.nmo[y] - self.nocc[y]
            m = None
            for (x, yy), tb in tbar.items():
                if yy != y or tb.size == 0:
                    continue
                mx = lib.dot(self._bov(x), tb.reshape(-1, no * nv))
                m = mx if m is None else m + mx
            if m is None:
                continue
            c0 -= weight * numpy.dot(self._bov(y).ravel(), m.ravel())
            m = m.reshape(-1, no, nv)
            c1[y][:, :no] += 2 * lib.einsum('Qpl,Qkl->pk', self._bgv[y], m)
            c1[y][:, no:] -= 2 * lib.einsum('Qkp,Qkl->pl', self._bog[y], m)
        return c0, c1


def make_df(mp, auxbasis=None):
    '''Density-fitting object for the correlation part (RI-C auxiliary basis
    by default, as in :mod:`pyscf.mp.dfmp2`).'''
    with_df = df.DF(mp.mol)
    with_df.max_memory = mp.max_memory
    with_df.stdout = mp.stdout
    with_df.verbose = mp.verbose
    if auxbasis is None:
        auxbasis = df.make_auxbasis(mp.mol, mp2fit=True)
    with_df.auxbasis = auxbasis
    return with_df
