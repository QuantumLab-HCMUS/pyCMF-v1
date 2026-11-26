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
Self-consistent driver shared by the one-body correlated methods.

At every cycle the method object builds a correlated Fock matrix from the
current orbitals and orbital energies (``mp.build``); the Fock matrix is
extrapolated by DIIS and diagonalised, and the new orbitals are occupied
according to ``mp.occupation``.  The driver only handles lists of channels.
A channel is a spin channel for molecules and an (irreducible) k-point for
periodic systems, so the same loop serves restricted, unrestricted and k-point
references.
'''

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger

from pycmf.core import occupation


def _diagonalize(fock_ao, mo_coeff):
    mo_energy, mo_new = [], []
    for f, c in zip(fock_ao, mo_coeff):
        e, u = scipy.linalg.eigh(c.conj().T.dot(f).dot(c))
        mo_energy.append(e)
        mo_new.append(c.dot(u))
    return mo_energy, mo_new


def _diis_error(fock_ao, dm, s1e, mo_basis):
    '''Commutator [F, D] projected on the space of the orbitals.'''
    errs = []
    for f, d, s, c in zip(fock_ao, dm, s1e, mo_basis):
        sdf = c.conj().T.dot(s).dot(d).dot(f).dot(c)
        errs.append((sdf.conj().T - sdf).ravel())
    return numpy.hstack(errs)


def kernel(mp, mo_energy, mo_coeff, nocc, verbose=None):
    '''Iterate the correlated Fock equations to self-consistency.

    Args:
        mp : method object providing ``build`` and the convergence attributes
        mo_energy, mo_coeff : lists (one entry per channel) of orbital
            energies and coefficients, occupied orbitals first
        nocc : list of occupation numbers

    Returns:
        converged, e_tot, mo_energy, mo_coeff, last intermediates
    '''
    log = logger.new_logger(mp, verbose)
    cput0 = cput1 = (logger.process_clock(), logger.perf_counter())
    nchan = len(mo_coeff)
    # one overlap matrix per channel (k-point methods provide their own list)
    s1e = getattr(mp, 'get_ovlp', mp._scf.get_ovlp)()
    if isinstance(s1e, numpy.ndarray) and s1e.ndim == 2:
        s1e = [s1e] * nchan
    conv_tol = mp.conv_tol
    conv_tol_grad = mp.conv_tol_grad
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)

    adiis = None
    if mp.diis:
        adiis = lib.diis.DIIS()
        adiis.space = mp.diis_space
        adiis.verbose = log.verbose
        adiis.stdout = mp.stdout
    # fixed orthonormal basis for the DIIS error vectors
    mo_diis = [c.copy() for c in mo_coeff]
    scheme = mp.occupation
    ref_occ = [c[:, :n].copy() for c, n in zip(mo_coeff, nocc)]

    converged = False
    e_tot = 0.
    im = None
    for cycle in range(mp.max_cycle):
        im = mp.build(mo_energy, mo_coeff, nocc)
        e_last, e_tot = e_tot, im.e_tot
        de = e_tot - e_last
        norm_g = max(numpy.linalg.norm(f[:n, n:]) for f, n in zip(im.fock_mo, nocc))
        log.info('cycle= %d E= %.15g  delta_E= %4.3g  |f_ov|= %4.3g',
                 cycle + 1, e_tot, de, norm_g)
        if scheme != 'aufbau':
            lost = sum(occupation.lost_overlap(c, n, r, s)
                       for c, n, r, s in zip(mo_coeff, nocc, ref_occ, s1e))
            log.debug('  electrons outside the reference occupied space: %.4f', lost)
        cput1 = log.timer_debug1('cycle %d' % (cycle + 1), *cput1)
        if cycle > 0 and abs(de) < conv_tol and norm_g < conv_tol_grad:
            converged = True
            break
        if cycle == mp.max_cycle - 1:
            # keep the orbitals that belong to the last energy evaluation
            break

        fock = im.fock_ao
        if adiis is not None and cycle >= mp.diis_start_cycle:
            fock = list(adiis.update(numpy.array(fock),
                                     xerr=_diis_error(fock, im.dm_chan, s1e, mo_diis)))
        if scheme == 'mom':
            ref_occ = [c[:, :n].copy() for c, n in zip(mo_coeff, nocc)]
        # Diagonalise in the space spanned by the current orbitals, so that
        # truncated or linearly dependent orbital spaces are preserved.
        mo_energy, mo_coeff = _diagonalize(fock, mo_coeff)
        for s in range(nchan):
            mo_energy[s], mo_coeff[s] = occupation.select(
                mo_energy[s], mo_coeff[s], nocc[s], scheme, ref_occ[s], s1e[s])

    log.timer(mp.__class__.__name__, *cput0)
    return converged, e_tot, mo_energy, mo_coeff, im
