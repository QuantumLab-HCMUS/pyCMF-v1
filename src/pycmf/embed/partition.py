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
Partition of the occupied orbitals of a system into an active subsystem and
its environment (SPADE).

Reference:
    D. Claudino and N. J. Mayhall, J. Chem. Theory Comput. 15, 1053 (2019)
'''

import numpy
import scipy.linalg


def atom_aos(mol, atoms):
    '''AO indices that belong to the given atoms.'''
    aoslice = mol.aoslice_by_atom()
    return numpy.hstack([numpy.arange(aoslice[i][2], aoslice[i][3]) for i in atoms])


def spade_partition(mol, s1e, mo_occ, atoms, return_sigma=False):
    '''Split occupied orbitals into subsystem and environment sets.

    The orbitals are rotated by the right singular vectors of their
    projection on the (symmetrically orthogonalised) AOs of the fragment and
    cut at the largest gap of the singular values.

    Args:
        s1e : AO overlap matrix
        mo_occ : occupied orbital coefficients of one spin, (nao, nocc)
        atoms : indices of the atoms of the active subsystem

    Returns:
        orbitals of the subsystem and of the environment, (nao, n_A) and
        (nao, n_B); with ``return_sigma`` also the singular values
    '''
    s_half = scipy.linalg.fractional_matrix_power(s1e, 0.5).real
    proj = s_half.dot(mo_occ)[atom_aos(mol, atoms)]
    _, sigma, vt = numpy.linalg.svd(proj, full_matrices=True)
    if len(sigma) == 0:
        n_act = 0
    elif sigma.min() > 0.99:
        n_act = mo_occ.shape[1]
    elif sigma.max() < 0.01:
        n_act = 0
    elif len(sigma) > 1:
        n_act = int(numpy.argmax(sigma[:-1] - sigma[1:])) + 1
    else:
        n_act = len(sigma)
    mo = mo_occ.dot(vt.conj().T)
    if return_sigma:
        return mo[:, :n_act], mo[:, n_act:], sigma
    return mo[:, :n_act], mo[:, n_act:]
