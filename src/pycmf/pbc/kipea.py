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
Second-order ionization-potential and electron-affinity corrections to the
bands of k-point OBMP2 (IPEA-KOBMP2).  EXPERIMENTAL.

The eigenvalues of the correlated Fock matrix are not removal or addition
energies.  For an occupied band h and a virtual band l at k-point k,

    IP(h,k) = -e(h,k) + 2 sum_{iaj} tbar[i,a,j,h] (ia|jh)
    EA(l,k) = -e(l,k) - 2 sum_{iab} tbar[i,a,l,b] (ia|lb)

with tbar = alpha t - beta t(exchange), t = conj(g) / (e_i + e_j - e_a - e_h)
for the IP and t = conj(g) / (e_i + e_l - e_a - e_b) for the EA, where
alpha = (c_ss + c_os)/2, beta = c_ss/2 as for the amplitudes of the potential,
and the sums run over the two free k-points.  The orbitals and energies are
the converged ones of KOBMP2.  Both corrections reduce the gap.

The corrections are static (no frequency dependence) and evaluated once,
after the self-consistent calculation.  For bands away from the gap the
denominators can vanish; only the band edges are meaningful.
'''

from pycmf.pbc import kqp


def kernel(mp, nvir=4, kpts_index=None):
    '''IPEA corrections for the bands of a converged KOBMP2 calculation.

    Args:
        mp : KOBMP2 object after ``kernel()``
        nvir : number of lowest virtual bands that are corrected
        kpts_index : indices (in the full mesh) of the k-points to treat;
            default: the irreducible k-points, with the result copied to the
            symmetry-equivalent ones

    Returns:
        ip, ea : lists over the k-points of the mesh.  ``ip[k]`` holds the
        IPs of all correlated occupied bands, ``ea[k]`` the EAs of the lowest
        ``nvir`` virtual bands (None for k-points that were not treated).

    The corrections are the static limit of :mod:`pycmf.pbc.kqp`, which also
    provides the frequency-dependent treatment.
    '''
    se = kqp.self_energy(mp, nvir, kpts_index, grid=[0.])
    mp._se = se
    ip = [None if s is None else -(s['e'] + s['B0'])[s['occ']] for s in se]
    ea = [None if s is None else -(s['e'] + s['A0'])[~s['occ']] for s in se]
    return ip, ea
