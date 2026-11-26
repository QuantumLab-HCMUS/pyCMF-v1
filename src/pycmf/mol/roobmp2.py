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
Restricted open-shell OBMP2 (RO-OBMP2) for high-spin molecules.  EXPERIMENTAL.

One set of spatial orbitals is shared by both spins.  The amplitudes and the
correlation potentials v_alpha, v_beta are those of unrestricted OBMP2
evaluated with the common orbitals; the two correlated Fock matrices are then
combined into Roothaan's effective Fock matrix, whose eigenvectors define the
next orbitals.  The amplitude denominators use the diagonals of the two spin
Fock matrices in the common orbital basis.

The determinant is a spin eigenfunction by construction.  No formal
derivation of the method from restricted open-shell perturbation theory is
available yet, and it has not been benchmarked.
'''

import numpy
import scipy.linalg
from pyscf import lib, scf
from pyscf.lib import logger
from pyscf.scf import rohf as pyscf_rohf

from pycmf.core import status
from pycmf.mol.obmp2 import UOBMP2


def _fill_occ(e_eff, e_alpha, ncore, nopen):
    '''Doubly occupy the lowest orbitals of the effective Fock matrix; singly
    occupy the lowest remaining ones according to the alpha energies.'''
    order = numpy.argsort(e_eff, kind='stable')
    rest = order[ncore:]
    open_idx = rest[numpy.argsort(e_alpha[rest], kind='stable')[:nopen]]
    vir = numpy.setdiff1d(rest, open_idx)
    return numpy.hstack((order[:ncore], numpy.sort(open_idx), vir))


def kernel(mp, verbose=None):
    log = logger.new_logger(mp, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    mf = mp._rohf
    mo_occ = numpy.asarray(mp.mo_occ)
    ncore = int(numpy.count_nonzero(mo_occ == 2))
    nopen = int(numpy.count_nonzero(mo_occ == 1))
    nocc = [ncore + nopen, ncore]
    idx = numpy.hstack([numpy.where(mo_occ == n)[0] for n in (2, 1, 0)])
    coeff = numpy.array(mp.mo_coeff)[:, idx]
    nmo = coeff.shape[1]

    e_a = getattr(mf.mo_energy, 'mo_ea', None)
    e_b = getattr(mf.mo_energy, 'mo_eb', None)
    if e_a is None:
        dm = mf.make_rdm1(mf.mo_coeff, mo_occ)
        focka, fockb = mf.get_hcore() + mf.get_veff(mp.mol, dm)
        e_a = numpy.einsum('pi,pq,qi->i', mf.mo_coeff, focka, mf.mo_coeff)
        e_b = numpy.einsum('pi,pq,qi->i', mf.mo_coeff, fockb, mf.mo_coeff)
    e_a, e_b = numpy.asarray(e_a)[idx], numpy.asarray(e_b)[idx]

    occ_a = numpy.where(numpy.arange(nmo) < nocc[0], 1., 0.)
    occ_b = numpy.where(numpy.arange(nmo) < nocc[1], 1., 0.)
    c, o, v = slice(0, ncore), slice(ncore, nocc[0]), slice(nocc[0], nmo)
    conv_tol_grad = mp.conv_tol_grad
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(mp.conv_tol)

    converged = False
    e_tot = 0.
    for cycle in range(mp.max_cycle):
        im = mp.build([e_a, e_b], [coeff, coeff], nocc)
        e_last, e_tot = e_tot, im.e_tot
        fa, fb = im.fock_mo
        f_eff = pyscf_rohf.get_roothaan_fock(
            (fa, fb), (numpy.diag(occ_a), numpy.diag(occ_b)), numpy.eye(nmo))
        norm_g = numpy.sqrt(sum(numpy.linalg.norm(f_eff[x, y])**2
                                for x, y in ((c, o), (c, v), (o, v))))
        norm_e = max(abs(numpy.diag(fa) - e_a).max(), abs(numpy.diag(fb) - e_b).max())
        log.info('cycle= %d E= %.15g  delta_E= %4.3g  |f_eff|= %4.3g  |d eps|= %4.3g',
                 cycle + 1, e_tot, e_tot - e_last, norm_g, norm_e)
        if (cycle > 0 and abs(e_tot - e_last) < mp.conv_tol and norm_g < conv_tol_grad
                and norm_e < mp.conv_tol_denom):
            converged = True
            break
        e_eff, u = scipy.linalg.eigh(f_eff)
        order = _fill_occ(e_eff, numpy.einsum('pi,pq,qi->i', u, fa, u), ncore, nopen)
        u = u[:, order]
        e_a = numpy.einsum('pi,pq,qi->i', u, fa, u)
        e_b = numpy.einsum('pi,pq,qi->i', u, fb, u)
        coeff = coeff.dot(u)

    log.timer('ROOBMP2', *cput0)
    mo_energy = lib.tag_array(numpy.diag(f_eff).copy(), mo_ea=e_a, mo_eb=e_b)
    mo_occ = occ_a + occ_b
    return converged, e_tot, mo_energy, coeff, mo_occ, im


class ROOBMP2(UOBMP2):
    '''High-spin restricted open-shell OBMP2 (experimental).

    Additional attribute:
        conv_tol_denom : threshold on the difference between the amplitude
            denominators and the diagonals of the spin Fock matrices.

    Saved results:
        mo_coeff : common spatial orbitals, ordered [closed | open | virtual]
        mo_occ : occupations 2 / 1 / 0
        mo_energy : diagonal of the effective Fock matrix, tagged with the
            spin energies ``mo_ea`` and ``mo_eb``
    '''
    conv_tol_denom = 1e-6
    diis = False
    _keys = UOBMP2._keys | {'conv_tol_denom'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if isinstance(mf, scf.uhf.UHF) or not isinstance(mf, scf.rohf.ROHF):
            raise TypeError('ROOBMP2 needs a ROHF reference.')
        if mf.mol.spin < 0:
            raise NotImplementedError('Only alpha-majority references are supported.')
        status.experimental('ROOBMP2')
        self._rohf = mf
        UOBMP2.__init__(self, mf.to_uhf(), frozen)
        self.mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
        self.mo_occ = mf.mo_occ if mo_occ is None else mo_occ
        self.e_hf = mf.e_tot

    def kernel(self):
        if self.occupation != 'aufbau':
            raise NotImplementedError('State following is not available for ROOBMP2.')
        self.dump_flags()
        (self.converged, self.e_tot, self.mo_energy, self.mo_coeff, self.mo_occ,
         im) = kernel(self)
        self.e_ref = im.e_ref
        self.e_corr = self.e_tot - self.e_hf
        im.blocks = None
        self._last = im
        self._finalize()
        return self.e_tot

    def _channels(self):
        if self._last is None:
            raise RuntimeError('Run kernel() first.')
        e = self.mo_energy
        return [e.mo_ea, e.mo_eb], [self.mo_coeff, self.mo_coeff], list(self._last.nocc)
