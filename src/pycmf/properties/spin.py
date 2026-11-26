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

'''Spin expectation values for unrestricted references.'''

import numpy
from pyscf import lib


def spin_square(mp, correlated=False):
    '''<S^2> and multiplicity 2S+1.

    Without correlation the value of the OBMP2 determinant is returned.
    With ``correlated=True`` the first-order doubles correction is added,

        <S^2> = <S^2>_det - sum_{ia,jb} S_ib S_ja t[i,a,j,b],

    where (i,a) are alpha, (j,b) beta orbitals and S is the alpha-beta
    orbital overlap.
    '''
    mo_energy, mo_coeff, nocc = mp._channels()
    s1e = mp._scf.get_ovlp()
    na, nb = nocc
    ca, cb = mo_coeff
    ss, mult = mp._scf.spin_square((ca[:, :na], cb[:, :nb]), s1e)
    if correlated:
        sab = ca.conj().T.dot(s1e).dot(cb)
        s_ib = sab[:na, nb:]
        s_ja = sab[na:, :nb].T
        im = mp.get_intermediates(with_amplitudes=True)
        for x, y, t, _ in im.blocks:
            if (x, y) == (0, 1):
                tmp = lib.einsum('ja,iajb->ib', s_ja, t)
                ss -= numpy.dot(s_ib.ravel(), tmp.ravel())
        mult = numpy.sqrt(ss + .25) * 2
    return ss, mult
