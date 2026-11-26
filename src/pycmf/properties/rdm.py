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
One-particle density matrices and dipole moments.

Two densities are available.  The "HF-like" density is the density of the
determinant built from the occupied OBMP2 orbitals.  The "MP2-like" density
adds the unrelaxed second-order correction evaluated with the converged
amplitudes (L. N. Tran, J. Phys. Chem. A 125, 9242 (2021)).
'''

import numpy
from pyscf.scf import hf


def make_rdm1_mo(mp, correlated=False):
    '''List of spin-channel density matrices in the MO basis (occupation
    numbers between 0 and 1).'''
    from pycmf.mol import bch
    mo_energy, mo_coeff, nocc = mp._channels()
    nchan = len(nocc)
    dms = []
    for c, n in zip(mo_coeff, nocc):
        d = numpy.zeros((c.shape[1],) * 2)
        d[numpy.diag_indices(n)] = 1.
        dms.append(d)
    if not correlated:
        return dms

    im = mp.get_intermediates()
    r = im.rdm
    if r is None:
        im = mp.get_intermediates(with_amplitudes=True)
        r = bch.density_intermediates(im.blocks, nchan)
    r_occ, r_vir = r
    for s, n in enumerate(nocc):
        if isinstance(r_occ[s], numpy.ndarray):
            dms[s][:n, :n] -= (r_occ[s] + r_occ[s].T) * .5
            dms[s][n:, n:] += (r_vir[s] + r_vir[s].T) * .5
    return dms


def make_rdm1(mp, ao_repr=True, correlated=False):
    '''Density matrix in the convention of the mean-field object: the total
    density for a restricted reference, (alpha, beta) otherwise.'''
    dms = make_rdm1_mo(mp, correlated)
    if ao_repr:
        mo_coeff = mp._channels()[1]
        dms = [c.dot(d).dot(c.conj().T) for c, d in zip(mo_coeff, dms)]
    return mp._pack([d * mp._occ for d in dms])


def dip_moment(mp, correlated=False, unit='Debye', verbose=None):
    '''Dipole moment as an expectation value over the chosen density.'''
    dm = mp.make_rdm1(ao_repr=True, correlated=correlated)
    dm = numpy.asarray(dm)
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    if verbose is None:
        verbose = mp.verbose
    return hf.dip_moment(mp.mol, dm, unit=unit, verbose=verbose)
