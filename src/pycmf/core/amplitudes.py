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
First-order doubles amplitudes of OBMP2.

Amplitudes are stored in chemists' order, ``t[i,a,j,b]``, as a list of spin
blocks ``(x, y, t, tbar)``.  ``x`` and ``y`` are the spin channels of the
pairs (i,a) and (j,b), ``t`` is the plain amplitude and ``tbar`` the
antisymmetrised one:

    t[i,a,j,b]    = c (ia|jb) / (e_i - e_a + e_j - e_b - shift)
    tbar[i,a,j,b] = t[i,a,j,b] - t[i,b,j,a]     (same spin)
    tbar          = t                           (opposite spin)

with c = c_ss for same-spin and c = c_os for opposite-spin pairs.  OBMP2 uses
c_ss = c_os = 1; O2BMP2 uses c_ss = 0 and an empirical c_os.  A block whose
scaling factor is zero is not built at all.

A restricted reference is described by a single channel and the two blocks
(same spin, opposite spin); the factor of two from the spin-flipped partners
is carried by the caller.
'''



def _denominator(e_x, e_y, shift):
    # e_x[i,a] = e_i - e_a (negative for an Aufbau determinant)
    return e_x[:, :, None, None] + e_y[None, None, :, :] - shift


def make_blocks(ints, mo_energy, nocc, c_os=1., c_ss=1., shift=0.):
    '''Build the list of amplitude blocks.

    Args:
        ints : integral backend providing ``ovov(x, y)``
        mo_energy : list of orbital energies, one array per spin channel
        nocc : list of occupation numbers per channel
        shift : constant added to the (positive) energy gap in the denominator
    '''
    nchan = len(mo_energy)
    e_ia = [e[:n, None] - e[None, n:] for e, n in zip(mo_energy, nocc)]
    blocks = []
    if nchan == 1:
        if c_ss == 0 and c_os == 0:
            return blocks
        t = ints.ovov(0, 0)
        t /= _denominator(e_ia[0], e_ia[0], shift)
        if c_ss != 0:
            tss = t * c_ss
            blocks.append((0, 0, tss, tss - tss.transpose(0, 3, 2, 1)))
        if c_os != 0:
            tos = t * c_os if c_os != 1 else t
            blocks.append((0, 0, tos, tos))
        return blocks

    if c_ss != 0:
        for x in range(nchan):
            t = ints.ovov(x, x)
            t /= _denominator(e_ia[x], e_ia[x], shift)
            if c_ss != 1:
                t *= c_ss
            blocks.append((x, x, t, t - t.transpose(0, 3, 2, 1)))
    if c_os != 0:
        t = ints.ovov(0, 1)
        t /= _denominator(e_ia[0], e_ia[1], shift)
        if c_os != 1:
            t *= c_os
        blocks.append((0, 1, t, t))
        tT = t.transpose(2, 3, 0, 1)
        blocks.append((1, 0, tT, tT))
    return blocks


def restrict_to_active(blocks, occ_act, vir_act):
    '''Zero every amplitude that carries an index outside the active space.

    Args:
        occ_act, vir_act : per-channel boolean masks over the occupied and
            the virtual orbitals
    '''
    out = []
    for x, y, t, tb in blocks:
        mask = (occ_act[x][:, None, None, None] & vir_act[x][None, :, None, None] &
                occ_act[y][None, None, :, None] & vir_act[y][None, None, None, :])
        out.append((x, y, t * mask, tb * mask))
    return out
