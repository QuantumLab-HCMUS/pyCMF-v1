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
Starting determinants for state-specific excited states.

A single excitation hole -> particle of one spin is converged at the
Hartree-Fock level with PySCF's maximum overlap method (Gilbert, Besley and
Gill, J. Phys. Chem. A 112, 13164 (2008)) and then re-optimized with a
correlated Fock operator using ``occupation='imom'``::

    mf_es = excited.mom_excitation(mf_gs, hole=4, particle=5, spin=1)
    es = pycmf.OBMP2(mf_es, occupation='imom').run()

Orbital indices count from 0 in the orbital order of ``mf_gs``.
'''

import numpy
from pyscf import scf
from pyscf.lib import logger


def _uhf(mf):
    if isinstance(mf, scf.uhf.UHF):
        return mf
    return scf.addons.convert_to_uhf(mf)


def mom_excitation(mf_gs, hole, particle, spin=0, conv_tol=None, max_cycle=None):
    '''MOM-UHF solution with one electron of ``spin`` (0 alpha, 1 beta) moved
    from orbital ``hole`` to orbital ``particle`` of ``mf_gs``.

    ``mf_gs`` is not modified.  The convergence settings are those of
    ``mf_gs`` unless given.  A warning is printed if MOM-UHF does not converge.
    '''
    mf_gs = _uhf(mf_gs)
    mo_coeff = numpy.asarray(mf_gs.mo_coeff)
    occ = numpy.array(mf_gs.mo_occ, copy=True)
    nmo = occ.shape[1]
    if spin not in (0, 1):
        raise ValueError('spin must be 0 (alpha) or 1 (beta)')
    if not (0 <= hole < nmo and 0 <= particle < nmo):
        raise ValueError('Orbital index outside 0..%d' % (nmo - 1))
    if occ[spin, hole] < .5 or occ[spin, particle] > .5:
        raise ValueError('hole must be occupied and particle empty in the reference '
                         '(spin %d: occ[%d] = %g, occ[%d] = %g)'
                         % (spin, hole, occ[spin, hole], particle, occ[spin, particle]))
    occ[spin, hole], occ[spin, particle] = 0, 1
    mf = scf.addons.mom_occ(scf.UHF(mf_gs.mol), mo_coeff, occ)
    mf.conv_tol = mf_gs.conv_tol if conv_tol is None else conv_tol
    mf.max_cycle = mf_gs.max_cycle if max_cycle is None else max_cycle
    mf.verbose = mf_gs.verbose
    mf.kernel(mf.make_rdm1(mo_coeff, occ))
    if not mf.converged:
        logger.warn(mf, 'MOM-UHF for %d -> %d (spin %d) not converged', hole, particle, spin)
    return mf


def select_channels(mf_gs, nchannels=None):
    '''Single excitations hole -> particle of a closed-shell reference ordered
    by the Hartree-Fock orbital gap.

    The gap is averaged over the two spins; the hole must be occupied and the
    particle empty in both.  Ordering by the gap is a way to choose a trial
    space, not a prediction of the excitation energies.

    Returns:
        list of (hole, particle, gap in Hartree), at most ``nchannels`` long
    '''
    mf_gs = _uhf(mf_gs)
    e = numpy.asarray(mf_gs.mo_energy)
    occ = numpy.asarray(mf_gs.mo_occ)
    occidx = numpy.where(numpy.all(occ > .5, axis=0))[0]
    viridx = numpy.where(numpy.all(occ < .5, axis=0))[0]
    channels = [(int(i), int(a), float(numpy.mean(e[:, a] - e[:, i])))
                for i in occidx for a in viridx]
    channels = [c for c in channels if c[2] > 0]
    channels.sort(key=lambda c: (c[2], c[0], c[1]))
    if nchannels is not None:
        if nchannels < 1:
            raise ValueError('nchannels must be positive')
        channels = channels[:nchannels]
    return channels


def channel_label(hole, particle, spin):
    '''Label used by :class:`pycmf.mol.noci.NOCI`, e.g. ``h4_p5_b``.'''
    return 'h%d_p%d_%s' % (hole, particle, 'ab'[spin])
