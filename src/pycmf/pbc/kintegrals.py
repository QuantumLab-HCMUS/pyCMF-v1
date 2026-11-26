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
Density-fitted integrals for the k-point methods.

All two-electron integrals are assembled from three-index tensors of a
Gaussian density fitting object of PySCF (GDF or RSGDF),

    (p k1, q k2 | r k3, s k4) = sum_Q L^Q_pq(k1, k2) L^Q_rs(k3, k4),

so that no four-index transformation is needed.  Two kinds of tensors are
used by :mod:`pycmf.pbc.kbch`:

* "rows": L_ia(k, k') for one k and all k', occupied-virtual block.  They
  build the amplitudes.  Rows are kept in memory while they fit in
  ``max_memory`` and in a temporary HDF5 file otherwise.

* "right" tensors: L_jp(kj, kb) and L_pb(kj, kb) with one general index.  They
  are needed once per k-point pair, for the final contraction of the
  correlation potential, and are never stored.

The AO tensors do not change during the self-consistent iterations; they are
kept in memory as far as ``max_memory`` allows after the rows, and read from
the file of the density fitting object otherwise.
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import df


def as_gdf(mf, auxbasis=None, with_df=None):
    '''Density fitting object for the correlation part.

    The object of the mean field is reused if it is a Gaussian density fitting
    (GDF, RSGDF); otherwise (FFTDF, AFTDF, MDF) a GDF object is built for all
    pairs of k-points.
    '''
    if with_df is not None:
        return with_df
    mfdf = getattr(mf, 'with_df', None)
    usable = isinstance(mfdf, df.GDF) and not isinstance(mfdf, df.MDF)
    if usable and auxbasis is None and not getattr(mfdf, '_j_only', False):
        return mfdf
    mydf = df.GDF(mf.cell, numpy.asarray(mf.kpts).reshape(-1, 3))
    if auxbasis is not None:
        mydf.auxbasis = auxbasis
    elif usable:
        mydf.auxbasis = mfdf.auxbasis
    return mydf


class KDFInts:
    '''Three-index integrals in the basis of the current orbitals.

    Args:
        with_df : GDF object
        kpts : k-points of the full Brillouin zone
        max_memory : memory (MB) available for the rows
    '''

    def __init__(self, with_df, kpts, max_memory=4000, verbose=None, stdout=None):
        self.with_df = with_df
        self.kpts = numpy.asarray(kpts).reshape(-1, 3)
        self.nkpts = len(self.kpts)
        self.max_memory = max_memory
        self.verbose = verbose if verbose is not None else with_df.verbose
        self.stdout = stdout if stdout is not None else with_df.stdout
        if with_df._cderi is None:
            with_df.build()
        self.mo_coeff = None
        self._rows = {}
        self._h5 = None
        self._incore = True
        self._ao = {}
        self._ao_mb = 0.
        self._ao_limit = max_memory * .3
        #: keep the (ov) tensors in single precision (half the memory; they are
        #: converted back for every use, so that all arithmetic is in double precision)
        self.single_precision = False
        self.t_ao = 0.
        self.t_mo = 0.

    def update(self, mo_coeff, occ, vir, nrows=None, nocc=None):
        '''Set the orbitals.

        Args:
            mo_coeff : list of orbital coefficients (full zone)
            occ, vir : lists of index arrays of the correlated occupied and
                virtual orbitals of every k-point
            nrows : number of rows that will be requested (to decide whether
                they fit in memory)
            nocc : number of occupied orbitals of every k-point (frozen ones
                included); needed for :meth:`build_rows` with the exchange
                matrices
        '''
        self.mo_coeff = [numpy.asarray(c, dtype=numpy.complex128) for c in mo_coeff]
        self.occ = occ
        self.vir = vir
        self._co = [c[:, o] for c, o in zip(self.mo_coeff, occ)]
        self._cv = [c[:, v] for c, v in zip(self.mo_coeff, vir)]
        self._nocc = None if nocc is None else [int(n) for n in nocc]
        self._rows = {}
        self._h5 = None
        if nrows is None:
            nrows = self.nkpts
        nov = max(len(o) for o in occ) * max(len(v) for v in vir)
        naux = self.with_df.get_naoaux()
        need = nrows * self.nkpts * naux * nov * (8e-6 if self.single_precision else 16e-6)
        # The rows are used once per irreducible k-point and cycle, the AO
        # tensors once per cycle: memory goes to the rows first.  The AO
        # tensors are cached only if at least half of them fit.
        self._incore = need < self.max_memory * .5
        self._ao_limit = self.max_memory * .8 - need if self._incore else self.max_memory * .3
        nao = self.mo_coeff[0].shape[0]
        if self._ao_limit < .5 * self.nkpts**2 * naux * nao**2 * 16e-6 and not self._ao:
            self._ao_limit = 0.
        logger.debug(self, 'three-index (ov) integrals: %.0f MB for %d rows, %s',
                     need, nrows, 'in memory' if self._incore else 'in a temporary file')
        if not self._incore:
            self._h5 = lib.H5TmpFile()
        return self

    def _ao_pair(self, k1, k2):
        '''L^Q_{mu nu}(k1, k2) as a complex (naux, nao, nao) array.'''
        if (k1, k2) in self._ao:
            return self._ao[k1, k2]
        t0 = logger.perf_counter()
        nao = self.mo_coeff[0].shape[0]
        out = []
        kpair = self.kpts[[k1, k2]]
        for lr, li, sign in self.with_df.sr_loop(kpair, compact=False):
            if sign != 1:
                raise NotImplementedError('density fitting with a non-positive metric')
            out.append(lr + 1j * li)
        out = out[0] if len(out) == 1 else numpy.vstack(out)
        out = out.reshape(-1, nao, nao)
        self.t_ao += logger.perf_counter() - t0
        if self._ao_mb + out.nbytes * 1e-6 < self._ao_limit:
            self._ao[k1, k2] = out
            self._ao_mb += out.nbytes * 1e-6
        return out

    def _half(self, lpq, k2, cols=None):
        c = numpy.ascontiguousarray(self.mo_coeff[k2] if cols is None else cols)
        naux, nao = lpq.shape[:2]
        if c.shape[1] == 0:
            return numpy.zeros((naux, nao, 0), dtype=numpy.complex128)
        return lib.dot(lpq.reshape(naux * nao, nao), c).reshape(naux, nao, c.shape[1])

    @staticmethod
    def _left(c, half):
        '''sum_p conj(c[p,i]) half[L,p,q] -> [L,i,q]'''
        naux, nao, ncol = half.shape
        if c.shape[1] == 0 or ncol == 0:
            return numpy.zeros((naux, c.shape[1], ncol), dtype=numpy.complex128)
        return lib.einsum('pi,Lpq->Liq', c.conj(), half)

    def row(self, k):
        '''List over k' of L^Q_ia(k, k') with shape (naux, nocc_k, nvir_k').'''
        if k in self._rows:
            if self.single_precision:
                return [x.astype(numpy.complex128) for x in self._rows[k]]
            return self._rows[k]
        if self._h5 is not None and str(k) in self._h5:
            grp = self._h5[str(k)]
            return [grp[str(ka)][()].astype(numpy.complex128) for ka in range(self.nkpts)]
        out = []
        for ka in range(self.nkpts):
            lpq = self._ao_pair(k, ka)
            t0 = logger.perf_counter()
            half = self._half(lpq, ka, self._cv[ka])
            out.append(self._left(self._co[k], half))
            self.t_mo += logger.perf_counter() - t0
        self._store(k, out)
        return out

    def build_rows(self, with_k=False):
        '''Compute and store all rows in one pass over the AO tensors.

        With ``with_k`` the exchange matrices of the closed-shell determinant
        of the occupied orbitals,

            K_{mu nu}(k) = 2/Nk sum_{k'} sum_{Q,i} L^Q_{mu i}(k,k') conj(L^Q_{nu i}(k,k')),

        are accumulated from the same AO tensors and returned (AO basis, the
        convention of PySCF, without the correction for the divergence of the
        exchange); otherwise None.  The integrals of a large mesh do not fit
        in the file cache, so that every pass over them is a pass over the
        disk: this saves the pass of the Hartree-Fock exchange.
        '''
        nk = self.nkpts
        vk = None
        if with_k:
            if self._nocc is None:
                raise ValueError('update() was called without nocc')
            nao = self.mo_coeff[0].shape[0]
            vk = numpy.zeros((nk, nao, nao), dtype=numpy.complex128)
            c_occ = [numpy.ascontiguousarray(c[:, :n]) for c, n in zip(self.mo_coeff, self._nocc)]
        for k in range(nk):
            done = k in self._rows or (self._h5 is not None and str(k) in self._h5)
            if done and not with_k:
                continue
            out = []
            for ka in range(nk):
                lpq = self._ao_pair(k, ka)
                t0 = logger.perf_counter()
                if not done:
                    half = self._half(lpq, ka, self._cv[ka])
                    out.append(self._left(self._co[k], half))
                if with_k and c_occ[ka].shape[1]:
                    half = self._half(lpq, ka, c_occ[ka])
                    naux, nao, nocc = half.shape
                    half = half.transpose(1, 0, 2).reshape(nao, naux * nocc)
                    vk[k] += lib.dot(half, half.conj().T)
                self.t_mo += logger.perf_counter() - t0
            if not done:
                self._store(k, out)
        if with_k:
            vk *= 2. / nk
        return vk

    def _store(self, k, out):
        store = [x.astype(numpy.complex64) for x in out] if self.single_precision else out
        if self._incore:
            self._rows[k] = store
        else:
            grp = self._h5.create_group(str(k))
            for ka, x in enumerate(store):
                grp[str(ka)] = x

    def ov(self, k1, k2):
        '''L^Q_ia(k1, k2) for one pair of k-points.'''
        if k1 in self._rows:
            return self._rows[k1][k2].astype(numpy.complex128, copy=False)
        if self._h5 is not None and str(k1) in self._h5:
            return self._h5[str(k1)][str(k2)][()].astype(numpy.complex128)
        return self.row(k1)[k2]

    def block(self, k1, k2, c1, c2):
        '''sum_{mu nu} conj(c1[mu,p]) L^Q_{mu nu}(k1, k2) c2[nu,q] -> [Q,p,q]'''
        lpq = self._ao_pair(k1, k2)
        t0 = logger.perf_counter()
        out = self._left(c1, self._half(lpq, k2, c2))
        self.t_mo += logger.perf_counter() - t0
        return out

    def right(self, kj, kb):
        '''L^Q_jp(kj, kb) and L^Q_pb(kj, kb): correlated occupied j, correlated
        virtual b and a general orbital index p.'''
        lpq = self._ao_pair(kj, kb)
        t0 = logger.perf_counter()
        half = self._half(lpq, kb)
        l_og = self._left(self._co[kj], half)
        l_gv = self._left(self.mo_coeff[kj], half[:, :, self.vir[kb]])
        self.t_mo += logger.perf_counter() - t0
        return l_og, l_gv
