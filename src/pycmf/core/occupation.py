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
Orbital-occupation schemes used after each diagonalization of the correlated
Fock matrix.

``aufbau``
    The lowest ``nocc`` orbitals are occupied (ground state).
``mom``
    Maximum overlap method: occupy the orbitals with the largest projection
    on the occupied space of the previous iteration
    (Gilbert, Besley, Gill, J. Phys. Chem. A 112, 13164 (2008)).
``imom``
    Initial maximum overlap method: the projection is taken on the occupied
    space of the starting determinant, which avoids drifting away from the
    target state (Barca, Gilbert, Gill, J. Chem. Theory Comput. 14, 1501 (2018)).
'''

import numpy

SCHEMES = ('aufbau', 'mom', 'imom')


def sort_by_occupation(mo_energy, mo_coeff, mo_occ):
    '''Reorder orbitals so that the occupied ones come first.

    The relative order inside the occupied and the virtual sets is kept, so
    that an Aufbau determinant is returned unchanged and a non-Aufbau
    (MOM-SCF) determinant becomes [occupied | virtual].
    '''
    occidx = numpy.where(mo_occ > 0)[0]
    viridx = numpy.where(mo_occ == 0)[0]
    idx = numpy.hstack((occidx, viridx))
    return mo_energy[idx], mo_coeff[:, idx], len(occidx)


def overlap_weights(mo_coeff, ref_occ, s1e):
    '''Squared norm of the projection of each orbital on span(ref_occ).'''
    proj = ref_occ.conj().T.dot(s1e).dot(mo_coeff)
    return numpy.einsum('ij,ij->j', proj.conj(), proj).real


def select(mo_energy, mo_coeff, nocc, scheme='aufbau', ref_occ=None, s1e=None):
    '''Return (mo_energy, mo_coeff) ordered as [occupied | virtual].

    Args:
        mo_energy, mo_coeff : eigenpairs of the Fock matrix, ascending energy
        nocc : number of occupied orbitals
        scheme : one of ``SCHEMES``
        ref_occ : reference occupied orbitals (AO basis) for mom / imom
        s1e : AO overlap matrix
    '''
    if scheme == 'aufbau' or nocc == 0:
        return mo_energy, mo_coeff
    if scheme not in SCHEMES:
        raise ValueError('Unknown occupation scheme %r' % (scheme,))
    weights = overlap_weights(mo_coeff, ref_occ, s1e)
    occidx = numpy.sort(numpy.argsort(-weights, kind='stable')[:nocc])
    mask = numpy.zeros(mo_energy.size, dtype=bool)
    mask[occidx] = True
    idx = numpy.hstack((occidx, numpy.where(~mask)[0]))
    return mo_energy[idx], mo_coeff[:, idx]


def lost_overlap(mo_coeff, nocc, ref_occ, s1e):
    '''Number of electrons of the reference determinant that are not
    recovered by the current occupied space (0 = same space).'''
    if nocc == 0:
        return 0.
    return nocc - overlap_weights(mo_coeff[:, :nocc], ref_occ, s1e).sum()
