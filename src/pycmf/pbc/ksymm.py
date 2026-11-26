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
k-point symmetry for one-body correlated methods.

The orbitals are kept at the irreducible k-points only.  Two operations are
needed to work with them:

* ``expand``: orbitals of the full Brillouin zone, obtained by applying the
  space-group operation (and complex conjugation for time reversal) that maps
  the irreducible representative on each k-point.  In this gauge the matrix of
  a totally symmetric one-body operator is the same at symmetry-related
  k-points (complex conjugated under time reversal).

* ``symmetrize``: if only a symmetry-reduced part of the terms of a totally
  symmetric operator has been summed, with every term weighted by the size of
  the star of its reduced index, the full operator is the group average

      v = 1/|G| sum_g  g^-1 P g

  of the partial sum P.  For a target k-point this needs P at all the k-points
  g.k and the matrices W that express the rotated orbitals g.phi(k) in the
  orbitals stored at g.k.

Symmetry detection and the rotation of atomic orbitals are those of PySCF
(:mod:`pyscf.pbc.lib.kpts` and :mod:`pyscf.pbc.symm`).
'''

import numpy
from pyscf.pbc.lib import kpts as libkpts
from pyscf.pbc.symm import symmetry as pbcsymm


class KSymmetry:
    '''Irreducible k-points of a mesh and the operations that relate them.

    Args:
        cell : PySCF cell
        kpts : (nkpts, 3) array, k-points of the full Brillouin zone
        space_group : use the (symmorphic) space-group operations
        time_reversal : use time-reversal symmetry

    Attributes:
        nkpts, nibz : number of k-points in the full zone and irreducible ones
        ibz2bz : index in the full zone of every irreducible k-point
        bz2ibz : irreducible representative of every k-point
        star : number of k-points equivalent to every irreducible k-point
        conj : True for the k-points reached with time reversal
    '''

    def __init__(self, cell, kpts, space_group=True, time_reversal=True):
        self.cell = cell
        self.kpts = numpy.asarray(kpts).reshape(-1, 3)
        kd = libkpts.KPoints(cell, self.kpts)
        kd.verbose = 0
        kd.build(space_group_symmetry=bool(space_group),
                 time_reversal_symmetry=bool(time_reversal))
        self.kd = kd
        self.nkpts = kd.nkpts
        self.nibz = kd.nkpts_ibz
        self.ibz2bz = numpy.asarray(kd.ibz2bz)
        self.bz2ibz = numpy.asarray(kd.bz2ibz)
        self.star = numpy.bincount(self.bz2ibz, minlength=self.nibz)
        self.conj = numpy.asarray(kd.time_reversal_symm_bz, dtype=bool)
        # operations (space group, then space group times time reversal) that
        # map the k-mesh on itself
        k2opk = numpy.asarray(kd.k2opk)
        self._k2opk = k2opk
        self._ops = [s for s in range(k2opk.shape[1]) if -1 not in k2opk[:, s]]
        self.trivial = self.nibz == self.nkpts
        self._rot = {}

    @property
    def nops(self):
        return len(self._ops)

    def _ao_rotation(self, k, s):
        '''AO matrix of operation s applied to Bloch functions of k-point k
        (without the complex conjugation of time reversal).'''
        if (k, s) not in self._rot:
            kd = self.kd
            iop = s % kd.nop
            op = kd.ops[iop]
            nao = self.cell.nao_nr()
            if op.is_eye:
                self._rot[k, s] = None
            else:
                self._rot[k, s] = pbcsymm.transform_mo_coeff(
                    self.cell, kd.kpts_scaled[k], numpy.eye(nao, dtype=numpy.complex128),
                    op, kd.Dmats[iop])
        return self._rot[k, s]

    def _rotated(self, k, s, mo_k):
        '''Orbitals of k-point k after operation s; they live at k2opk[k,s].'''
        mat = self._ao_rotation(k, s)
        out = mo_k if mat is None else mat.dot(mo_k)
        tr = s >= self.kd.nop
        return (out.conj() if tr else out), tr

    def expand(self, mo_ibz):
        '''Orbital coefficients of the full zone from the irreducible ones.'''
        if self.trivial:
            return list(mo_ibz)
        kd = self.kd
        out = []
        for k, i in enumerate(self.bz2ibz):
            s = kd.stars_ops_bz[k] + kd.nop * int(kd.time_reversal_symm_bz[k])
            out.append(self._rotated(self.ibz2bz[i], s, numpy.asarray(mo_ibz[i]))[0])
        return out

    def expand_matrix(self, mat_ibz):
        '''MO matrices of a totally symmetric operator in the full zone, in
        the gauge of :meth:`expand`.'''
        return [mat_ibz[i].conj() if self.conj[k] else mat_ibz[i]
                for k, i in enumerate(self.bz2ibz)]

    def expand_scalar(self, x_ibz):
        return [x_ibz[i] for i in self.bz2ibz]

    def symmetrize(self, mats, mo_coeff, s1e):
        '''Group average of partially summed operators.

        Args:
            mats : list of lists; ``mats[n][k]`` is the MO matrix at k-point k
                (full zone) of the n-th partial sum
            mo_coeff, s1e : orbitals and overlap matrices of the full zone

        Returns:
            list of lists; ``out[n][i]`` is the complete MO matrix at the
            i-th irreducible k-point, in the basis ``mo_coeff[ibz2bz[i]]``
        '''
        out = [[None] * self.nibz for _ in mats]
        if self.trivial:
            for n, m in enumerate(mats):
                out[n] = [m[k] for k in self.ibz2bz]
            return out
        norm = 1. / len(self._ops)
        for i, k in enumerate(self.ibz2bz):
            acc = [0] * len(mats)
            for s in self._ops:
                kp = self._k2opk[k, s]
                rot, tr = self._rotated(k, s, mo_coeff[k])
                w = mo_coeff[kp].conj().T.dot(s1e[kp]).dot(rot)
                err = abs(w.conj().T.dot(w) - numpy.eye(w.shape[1])).max()
                if err > 1e-6:
                    raise RuntimeError(
                        'The orbitals do not follow the symmetry of the crystal '
                        '(deviation %.2g).  Rerun with kpt_symmetry=False.' % err)
                for n, m in enumerate(mats):
                    x = w.conj().T.dot(m[kp]).dot(w)
                    acc[n] = acc[n] + (x.conj() if tr else x)
            for n in range(len(mats)):
                out[n][i] = acc[n] * norm
        return out
