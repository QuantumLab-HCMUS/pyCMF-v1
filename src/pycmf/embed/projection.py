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
Projection-based embedding of the one-body correlated methods in DFT.
EXPERIMENTAL.

The occupied orbitals of a Kohn-Sham calculation on the full system (the low
level, functional ``xc_env``) are partitioned into an active subsystem A and
an environment B (SPADE).  The subsystem is treated at the high level, OBMP2
or OBDH, in the presence of the embedding potential

    v_emb = v_eff^env[gamma_A + gamma_B] - v_eff^env[gamma_A] + mu S gamma_B S,

where the level-shift projector keeps the subsystem orbitals orthogonal to
the environment.  The virtual space can be truncated by concentric
localization.  The total energy is

    E = E_high[gamma_A'] + E^env[gamma_A' + gamma_B] - E^env[gamma_A']
        + mu tr(gamma_A' S gamma_B S),

with gamma_A' the relaxed density of the embedded subsystem and E^env, v^env
the Kohn-Sham energy and potential of the low-level functional.

Functionals.  The two levels are independent, as in wave-function-in-DFT
embedding: the environment and the embedding potential use ``xc_env``; the
high level is Hartree-Fock plus the OBMP2 correlation (EmbeddedOBMP2) or the
double hybrid of the subsystem, defined as in :mod:`pycmf.mol.obdh` by
``xc`` or (``ax``, ``ac``, ``xc_x``, ``xc_c``), B2PLYP by default
(EmbeddedOBDH).  Two limits are exact and are part of the test suite: with
all atoms in the subsystem the embedded energy is the energy of the high-level
method for the full system, and DFT-in-DFT embedding with the same functional
on both levels returns the Kohn-Sham energy of the full system.

References:
    F. R. Manby et al., J. Chem. Theory Comput. 8, 2564 (2012)
    D. Claudino and N. J. Mayhall, J. Chem. Theory Comput. 15, 1053 (2019)
    D. Claudino and N. J. Mayhall, J. Chem. Theory Comput. 15, 6085 (2019)
'''

import numpy
import scipy.linalg
from pyscf import dft, scf
from pyscf.lib import logger

from pycmf.core import occupation, scf_loop, status
from pycmf.embed import cl
from pycmf.embed.partition import atom_aos, spade_partition
from pycmf.mol.obmp2 import UOBMP2
from pycmf.mol.obdh import UOBDH

__all__ = ['EmbeddedOBMP2', 'EmbeddedOBDH', 'spade_partition', 'atom_aos']


class _EmbedMixin:
    '''Attributes:
        active_atoms : indices of the atoms of the embedded subsystem
        xc_env : functional of the low level (environment and embedding
            potential).  Default: the functional of the mean-field object if
            it is a Kohn-Sham one, otherwise 'pbe'.
        mu : level-shift parameter of the projector.  Default 1e6.
        use_cl, n_shells : truncate the virtual space to ``n_shells``
            concentric-localization shells

    Saved results:
        e_tot : embedded total energy
        e_env : Kohn-Sham energy of the full system at the low level
        e_sub : energy of the subsystem at the high level
        e_low_sub : energy of the subsystem at the low level (relaxed density)
        nelec_sub : (alpha, beta) electrons of the subsystem
        mo_energy, mo_coeff, mo_occ : orbitals of the embedded subsystem
    '''
    active_atoms = None
    xc_env = None
    mu = 1e6
    use_cl = False
    n_shells = 1
    _embed_keys = {'active_atoms', 'xc_env', 'mu', 'use_cl', 'n_shells', 'e_env', 'e_sub',
                   'e_low_sub', 'nelec_sub'}

    def _init_embed(self, mf):
        status.experimental(self.__class__.__name__)
        self._mf_full = mf
        self._ks_full = None
        self._v_emb = None
        self._dm_env = None
        self.e_env = None
        self.e_sub = None
        self.e_low_sub = None
        self.nelec_sub = None

    def get_xc_env(self):
        '''Functional of the low level.'''
        if self.xc_env is not None:
            return self.xc_env
        return getattr(self._mf_full, 'xc', None) or 'pbe'

    def _low_level(self):
        '''Converged unrestricted Kohn-Sham object of the full system.  The
        mean-field object given by the user is reused if it is a converged
        Kohn-Sham calculation with the functional ``xc_env``.'''
        mf = self._mf_full
        xc = self.get_xc_env()
        same = str(getattr(mf, 'xc', None)).upper().replace(' ', '') == xc.upper().replace(' ', '')
        if same and getattr(mf, 'converged', False):
            return mf if isinstance(mf, scf.uhf.UHF) else mf.to_uks()
        ks = dft.UKS(mf.mol)
        ks.xc = xc
        ks.verbose = self.verbose
        ks.stdout = self.stdout
        with_df = getattr(mf, 'with_df', None)
        if with_df:
            ks = ks.density_fit()
            ks.with_df = with_df
        ks.kernel()
        if not ks.converged:
            raise RuntimeError('Kohn-Sham calculation of the full system did not converge.')
        return ks

    def _embedded_scf(self, mol, h1e, v_emb, dm0, nelec):
        '''UHF of the subsystem in the embedding potential (starting orbitals
        of the correlated calculation).'''
        mol_emb = mol.copy()
        mol_emb.nelectron = sum(nelec)
        mol_emb.spin = nelec[0] - nelec[1]
        mf = scf.UHF(mol_emb)
        with_df = getattr(self._mf_full, 'with_df', None)
        if with_df:
            mf = mf.density_fit()
            mf.with_df = with_df
        mf.verbose = self.verbose
        mf.stdout = self.stdout
        mf.max_cycle = 200
        get_veff0 = mf.get_veff

        def get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, *args, **kwargs):
            if dm is None:
                dm = mf.make_rdm1()
            return numpy.asarray(get_veff0(mol_emb, numpy.asarray(dm))) + v_emb
        mf.get_veff = get_veff
        mf.get_hcore = lambda *args: h1e
        mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError('Embedded UHF did not converge.')
        return mf

    def _truncate_virtuals(self, mf, aos):
        '''Occupied + CL-truncated, canonicalised virtual orbitals.'''
        s1e = mf.get_ovlp()
        fock = mf.get_fock()
        nvir = [int(((o == 0) & (e < 1e5)).sum()) for o, e in zip(mf.mo_occ, mf.mo_energy)]
        n_span = min(len(aos), min(nvir))
        mo_energy, mo_coeff, nocc = [], [], []
        for s in range(2):
            occ = mf.mo_occ[s] > 0
            vir = (mf.mo_occ[s] == 0) & (mf.mo_energy[s] < 1e5)
            c_vir = cl.concentric_localization(mf.mo_coeff[s][:, vir], s1e, fock[s], aos,
                                               self.n_shells, n_span=n_span)
            e_vir, u = scipy.linalg.eigh(c_vir.conj().T.dot(fock[s]).dot(c_vir))
            mo_coeff.append(numpy.hstack((mf.mo_coeff[s][:, occ], c_vir.dot(u))))
            mo_energy.append(numpy.hstack((mf.mo_energy[s][occ], e_vir)))
            nocc.append(int(occ.sum()))
        return mo_energy, mo_coeff, nocc

    def kernel(self):
        log = logger.new_logger(self)
        if self.active_atoms is None or len(self.active_atoms) == 0:
            raise ValueError('active_atoms is not set.')
        mol = self._mf_full.mol
        s1e = self._mf_full.get_ovlp()

        # 1. low-level calculation on the full system and partition
        ks_full = self._ks_full = self._low_level()
        h1e = ks_full.get_hcore()
        part = [spade_partition(mol, s1e, c[:, o > 0], self.active_atoms)
                for c, o in zip(ks_full.mo_coeff, ks_full.mo_occ)]
        dm_a = numpy.array([c_a.dot(c_a.conj().T) for c_a, _ in part])
        dm_b = numpy.array([c_b.dot(c_b.conj().T) for _, c_b in part])
        nelec = self.nelec_sub = tuple(c_a.shape[1] for c_a, _ in part)
        log.info('Embedding in %s: %s active electrons (alpha, beta) on atoms %s',
                 self.get_xc_env(), nelec, list(self.active_atoms))
        if not 0 <= nelec[0] - nelec[1] <= mol.spin:
            log.warn('Unexpected spin of the embedded subsystem: %s', nelec)

        # 2. embedding potential
        proj_b = numpy.array([s1e.dot(d).dot(s1e) for d in dm_b])
        v_full = numpy.asarray(ks_full.get_veff(mol, ks_full.make_rdm1()))
        v_sub = numpy.asarray(ks_full.get_veff(mol, dm_a))
        self._v_emb = v_emb = v_full - v_sub + self.mu * proj_b
        self._dm_env = dm_b

        # 3. embedded mean field, optional truncation of the virtual space
        mf_emb = self._embedded_scf(mol, h1e, v_emb, dm_a, nelec)
        self._scf = self._hf = mf_emb
        self.mol = mf_emb.mol
        self._ints = self._ks = None
        if self.use_cl:
            mo_energy, mo_coeff, nocc = self._truncate_virtuals(
                mf_emb, atom_aos(mol, self.active_atoms))
        else:
            out = [occupation.sort_by_occupation(e, c, o) for e, c, o in
                   zip(mf_emb.mo_energy, mf_emb.mo_coeff, mf_emb.mo_occ)]
            mo_energy, mo_coeff, nocc = ([x[i] for x in out] for i in range(3))

        # 4. correlated calculation on the subsystem
        self.dump_flags()
        self.converged, _, mo_energy, mo_coeff, im = scf_loop.kernel(
            self, mo_energy, mo_coeff, nocc)
        self.mo_energy, self.mo_coeff = mo_energy, mo_coeff
        self.mo_occ = [numpy.where(numpy.arange(c.shape[1]) < n, 1., 0.)
                       for c, n in zip(mo_coeff, nocc)]
        dm_sub = numpy.array(im.dm_chan)

        # 5. energy
        self.e_low_sub = ks_full.energy_elec(dm_sub, h1e=h1e)[0] + mol.energy_nuc()
        e_low_full = ks_full.energy_tot(dm=dm_sub + dm_b)
        e_proj = self.mu * numpy.einsum('sij,sji->', dm_sub, proj_b)
        self.e_sub = self._subsystem_energy(im, dm_sub, h1e, mol)
        self.e_env = ks_full.e_tot
        self.e_tot = self.e_sub + e_low_full - self.e_low_sub + e_proj
        self.e_ref = im.e_ref
        log.info('E(high, A) = %.12g  E(low, A+B) - E(low, A) = %.12g  projector = %.3g',
                 self.e_sub, e_low_full - self.e_low_sub, e_proj)
        im.blocks = None
        self._last = im
        self._finalize()
        return self.e_tot

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'low-level functional xc_env = %s', self.get_xc_env())
        logger.info(self, 'active atoms = %s, mu = %g, CL shells = %s',
                    list(self.active_atoms), self.mu, self.n_shells if self.use_cl else 'all')
        return self

    def make_rdm1(self, ao_repr=True, correlated=False, with_env=True):
        '''Density matrix (alpha, beta).  With ``with_env`` (default) the
        density of the environment is added, which gives the density of the
        full system (AO basis only); otherwise that of the subsystem.'''
        dm = UOBMP2.make_rdm1(self, ao_repr, correlated)
        if with_env:
            if not ao_repr:
                raise ValueError('with_env needs ao_repr=True')
            dm = numpy.asarray(dm) + self._dm_env
        return dm


class EmbeddedOBMP2(_EmbedMixin, UOBMP2):
    '''OBMP2 embedded in DFT (experimental).

    Args:
        mf : mean-field object of the full system.  A converged Kohn-Sham
            object is used as the low-level calculation; otherwise it only
            provides the molecule and the density-fitting settings, and a
            Kohn-Sham calculation with ``xc_env`` is run.
        active_atoms : indices of the atoms of the subsystem
    '''
    _keys = UOBMP2._keys | _EmbedMixin._embed_keys

    def __init__(self, mf, active_atoms=None, **kwargs):
        UOBMP2.__init__(self, mf)
        self._init_embed(mf)
        self.active_atoms = active_atoms
        self.set(**kwargs)

    def _subsystem_energy(self, im, dm_sub, h1e, mol):
        hf = scf.UHF(mol)
        with_df = getattr(self._mf_full, 'with_df', None)
        if with_df:
            hf = hf.density_fit()
            hf.with_df = with_df
        return hf.energy_elec(dm_sub, h1e=h1e)[0] + mol.energy_nuc() + im.e_corr


class EmbeddedOBDH(_EmbedMixin, UOBDH):
    '''OBDH embedded in DFT (experimental).  Arguments as for
    :class:`EmbeddedOBMP2`; the double hybrid of the subsystem is chosen with
    ``xc`` or ``ax``, ``ac``, ``xc_x``, ``xc_c`` (default B2PLYP).'''
    _keys = UOBDH._keys | _EmbedMixin._embed_keys

    def __init__(self, mf, active_atoms=None, **kwargs):
        UOBDH.__init__(self, mf)
        self._init_embed(mf)
        self.active_atoms = active_atoms
        self.set(**kwargs)

    def get_ks(self):
        '''Kohn-Sham object of the high-level functional, on the integration
        grid of the low-level calculation.'''
        ks = UOBDH.get_ks(self)
        if self._ks_full is not None and self.grids_level is None:
            ks.grids = self._ks_full.grids
        return ks

    def get_reference(self, dm, h1e, vhf, fock_hf, e_hf):
        fock, e_ks, scale = UOBDH.get_reference(self, dm, h1e, vhf, fock_hf, e_hf)
        return [f + v for f, v in zip(fock, self._v_emb)], e_ks, scale

    def _subsystem_energy(self, im, dm_sub, h1e, mol):
        return im.e_tot
