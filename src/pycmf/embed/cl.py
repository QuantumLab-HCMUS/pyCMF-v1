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
Concentric localization (CL) of the virtual space of an embedded subsystem.

The virtual orbitals are ordered in shells of decreasing relevance for the
active fragment: shell 0 is the span of the virtual space projected on the
AOs of the fragment; shell n+1 collects the remaining orbitals that couple to
shell n through the Fock operator.  Keeping the first few shells truncates the
virtual space of the correlated calculation.

Reference:
    D. Claudino and N. J. Mayhall, J. Chem. Theory Comput. 15, 6085 (2019)
'''

import numpy
import scipy.linalg


def _split(vt, n_keep_mask):
    return vt[n_keep_mask].conj().T, vt[~n_keep_mask].conj().T


def shells(mo_vir, s1e, fock, active_aos, n_shells=-1, tol=1e-5, n_span=None):
    '''List of CL shells (AO coefficients).

    Args:
        mo_vir : virtual orbitals to be localized, (nao, nvir)
        fock : Fock matrix of the same spin, AO basis
        active_aos : AO indices of the active fragment
        n_shells : number of shells to build; -1 for all
        tol : singular-value threshold separating span and kernel
        n_span : fixed size of each shell (overrides ``tol``)
    '''
    nvir = mo_vir.shape[1]
    s_half = scipy.linalg.fractional_matrix_power(s1e, 0.5).real
    proj = s_half.dot(mo_vir)[active_aos]
    try:
        _, sigma, vt = scipy.linalg.svd(proj, full_matrices=True)
    except scipy.linalg.LinAlgError:
        _, sigma, vt = scipy.linalg.svd(proj, full_matrices=True, lapack_driver='gesvd')

    def mask(sigma, n):
        m = numpy.zeros(n, dtype=bool)
        if n_span is not None:
            m[:min(n_span, n)] = True
        else:
            m[:len(sigma)] = sigma > tol
        return m

    v_span, v_ker = _split(vt, mask(sigma, nvir))
    out = [mo_vir.dot(v_span)]
    c_ker = mo_vir.dot(v_ker)
    n_extra = nvir if n_shells == -1 else max(n_shells - 1, 0)
    for _ in range(n_extra):
        if c_ker.shape[1] == 0:
            break
        coupling = out[-1].conj().T.dot(fock).dot(c_ker)
        _, sigma, vt = scipy.linalg.svd(coupling, full_matrices=True)
        m = mask(sigma, c_ker.shape[1])
        if not m.any():
            break
        v_span, v_ker = _split(vt, m)
        out.append(c_ker.dot(v_span))
        c_ker = c_ker.dot(v_ker)
    return out


def concentric_localization(mo_vir, s1e, fock, active_aos, n_shells=1, tol=1e-5, n_span=None):
    '''Truncated virtual space spanned by the first ``n_shells`` CL shells.'''
    return numpy.hstack(shells(mo_vir, s1e, fock, active_aos, n_shells, tol, n_span))


def shell_sizes(mo_vir, s1e, fock, active_aos, tol=1e-5):
    '''Sizes of all CL shells, to choose ``n_shells`` before a calculation.'''
    return [c.shape[1] for c in shells(mo_vir, s1e, fock, active_aos, -1, tol)]
