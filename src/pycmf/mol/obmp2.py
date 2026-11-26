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
One-body second-order Møller-Plesset perturbation theory (OBMP2) and its
spin-opposite-scaled variant (O2BMP2) for molecules.

OBMP2 replaces the Hartree-Fock Fock operator by a correlated Fock operator,

    fbar_pq = f_pq + v_pq,

where v is the one-body MP2 correlation potential obtained from a canonical
transformation and a cumulant reduction.  The eigenvectors and eigenvalues of
fbar define the orbitals and the denominators of the next set of amplitudes;
the procedure is repeated until self-consistency.  The first cycle from HF
orbitals reproduces the conventional MP2 energy.

Simple usage::

    >>> from pyscf import gto, scf
    >>> import pycmf
    >>> mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1)
    >>> mf = scf.UHF(mol).run()
    >>> mp = pycmf.OBMP2(mf).run()
    >>> mp.e_tot, mp.spin_square()

References:
    T. N. Lan and T. Yanai, J. Chem. Phys. 138, 224108 (2013)
    L. N. Tran, J. Phys. Chem. A 125, 9242 (2021)
    N. T. Tran, H. T. Nguyen and L. N. Tran, J. Phys. Chem. A 128, 1543 (2024)  [O2BMP2]
    N. T. Tran and L. N. Tran, J. Chem. Phys. 162, 104109 (2025)  [state-specific]
'''

import numpy
from pyscf import lib, scf
from pyscf.lib import logger

from pycmf.core import amplitudes, occupation, scf_loop
from pycmf.mol import bch, integrals
from pycmf import properties


class Intermediates:
    '''Quantities of one cycle (all MO matrices in the basis of that cycle).

    Attributes:
        e_tot : total energy
        e_ref : energy of the reference functional (HF expectation value)
        e_corr : correlation energy, e_tot - e_ref
        dm_chan : list of AO density matrices per spin channel
        fock_ao, fock_mo : correlated Fock matrices, AO and MO basis
        fock_hf : HF Fock matrices in the MO basis
        v : correlation potential in the MO basis
        c0 : constant of the correlation potential
        rdm : intermediates of the MP2-like density
        blocks : amplitude blocks
    '''


class OBMP2Base(lib.StreamObject):
    '''Base class of the one-body MP2 methods.

    Attributes:
        conv_tol : float
            Convergence threshold of the energy.  Default 1e-8.
        conv_tol_grad : float
            Convergence threshold of the norm of the occupied-virtual block
            of the correlated Fock matrix.  Default sqrt(conv_tol).
        max_cycle : int
            Maximum number of cycles.  Default 100.
        bch_order : int
            2 (default): H + [H,A]_1 + 1/2 [[F,A],A]_1.
            1: H + 1/2 [H,A]_1, a cheaper variant.
        c_os, c_ss : float
            Scaling of the opposite-spin and same-spin amplitudes.
            (1, 1) is OBMP2; (c_os, 0) is O2BMP2.
        level_shift : float
            Constant added to the energy gap in the amplitude denominators.
        occupation : str
            'aufbau' (ground state), 'mom' or 'imom' (state-specific
            calculations started from a non-Aufbau determinant).
        diis, diis_space, diis_start_cycle :
            DIIS extrapolation of the correlated Fock matrix.
        with_df :
            Density-fitting object.  None means exact integrals.

    Saved results:
        converged : bool
        e_tot : total energy
        e_corr : e_tot - E(input mean field)
        e_ref : HF energy of the OBMP2 determinant
        mo_energy, mo_coeff, mo_occ : correlated orbitals, occupied first
    '''

    conv_tol = 1e-8
    conv_tol_grad = None
    max_cycle = 100
    bch_order = 2
    c_os = 1.
    c_ss = 1.
    level_shift = 0.
    occupation = 'aufbau'
    diis = True
    diis_space = 8
    diis_start_cycle = 1

    _keys = {'conv_tol', 'conv_tol_grad', 'max_cycle', 'bch_order', 'c_os', 'c_ss',
             'level_shift', 'occupation', 'diis', 'diis_space', 'diis_start_cycle',
             'mol', 'max_memory', 'frozen', 'with_df', 'mo_energy', 'mo_coeff',
             'mo_occ', 'converged', 'e_tot', 'e_corr', 'e_ref', 'e_hf'}

    # number of electrons per occupied spatial orbital of one spin channel
    _occ = None

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if frozen not in (None, 0):
            raise NotImplementedError('Frozen orbitals are not supported: all '
                                      'orbitals are relaxed by the correlated Fock operator.')
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.frozen = None
        self.with_df = getattr(mf, 'with_df', None) or None

        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
        self.mo_occ = mf.mo_occ if mo_occ is None else mo_occ
        self.converged = False
        self.e_hf = getattr(mf, 'e_tot', None)
        self.e_tot = None
        self.e_corr = None
        self.e_ref = None
        self._ints = None
        self._hf = None
        self._last = None

    # ---- spin handling: implemented by the restricted/unrestricted classes
    def _pack(self, chan):
        raise NotImplementedError

    def _unpack(self, x):
        raise NotImplementedError

    def _make_dm(self, mo_coeff, nocc):
        return [self._occ * c[:, :n].dot(c[:, :n].conj().T) for c, n in zip(mo_coeff, nocc)]

    # ---- building blocks
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('bch_order = %d', self.bch_order)
        log.info('c_os = %g  c_ss = %g', self.c_os, self.c_ss)
        log.info('level_shift = %g', self.level_shift)
        log.info('occupation = %s', self.occupation)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('DIIS = %s, space = %d', self.diis, self.diis_space)
        log.info('integrals = %s', 'density fitting' if self.with_df is not None else 'exact')
        log.info('max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        return self

    def density_fit(self, auxbasis=None, with_df=None):
        '''Use density fitting for the correlation potential.

        Without arguments, the DF object of the mean-field calculation is
        reused if there is one; otherwise an RI-C auxiliary basis is chosen.
        '''
        if with_df is not None:
            self.with_df = with_df
        elif auxbasis is not None or self.with_df is None:
            self.with_df = integrals.make_df(self, auxbasis)
        self._ints = None
        return self

    def get_hf(self):
        '''Hartree-Fock object used to build the HF Fock matrix (the input
        mean field itself unless it is a Kohn-Sham object).'''
        if self._hf is None:
            mf = self._scf
            self._hf = mf.to_hf() if hasattr(mf, 'xc') else mf
        return self._hf

    def ao2mo(self):
        '''Integral backend.'''
        if self._ints is None:
            if self.with_df is not None:
                self._ints = integrals.DFInts(self)
            else:
                self._ints = integrals.IncoreInts(self)
        return self._ints

    def get_reference(self, dm, h1e, vhf, fock_hf, e_hf):
        '''Reference Fock matrix (AO), reference energy and the weight of the
        correlation part.  Overloaded by the double-hybrid methods.'''
        return fock_hf, e_hf, 1.

    def make_amplitudes(self, mo_energy, mo_coeff, nocc, ints=None):
        if ints is None:
            ints = self.ao2mo().update(mo_coeff, nocc)
        return amplitudes.make_blocks(ints, mo_energy, nocc, self.c_os, self.c_ss,
                                      self.level_shift)

    def build(self, mo_energy, mo_coeff, nocc):
        '''Correlated Fock matrix and energy for a given set of orbitals.

        All arguments are lists over spin channels; orbitals are ordered as
        [occupied | virtual].
        '''
        hf = self.get_hf()
        mol = self.mol
        s1e = hf.get_ovlp()
        h1e = hf.get_hcore()
        dm_chan = self._make_dm(mo_coeff, nocc)
        dm = self._pack(dm_chan)
        vhf = hf.get_veff(mol, dm)
        e_hf = hf.energy_elec(dm, h1e, vhf)[0] + hf.energy_nuc()
        fock_hf_ao = self._unpack(h1e + vhf)
        fock_hf = [c.conj().T.dot(f).dot(c) for c, f in zip(mo_coeff, fock_hf_ao)]

        ints = self.ao2mo().update(mo_coeff, nocc)
        blocks = self.make_amplitudes(mo_energy, mo_coeff, nocc, ints)
        v, c0, rdm = bch.correlation_potential(
            ints, blocks, fock_hf, nocc, self.bch_order, weight=self._occ)
        e_c = c0 + self._occ * sum(numpy.trace(x[:n, :n]) for x, n in zip(v, nocc))

        fock_ref_ao, e_ref, scale = self.get_reference(dm, h1e, vhf, fock_hf_ao, e_hf)
        fock_ao, fock_mo = [], []
        for s, c in enumerate(mo_coeff):
            sc = s1e.dot(c)
            fock_ao.append(fock_ref_ao[s] + scale * sc.dot(v[s]).dot(sc.conj().T))
            fock_mo.append(c.conj().T.dot(fock_ref_ao[s]).dot(c) + scale * v[s])

        im = Intermediates()
        im.e_ref = e_ref
        im.e_corr = scale * e_c
        im.e_tot = e_ref + scale * e_c
        im.e_hf = e_hf
        im.dm_chan = dm_chan
        im.fock_ao = fock_ao
        im.fock_mo = fock_mo
        im.fock_hf = fock_hf
        im.v = v
        im.c0 = c0
        im.rdm = rdm
        im.blocks = blocks
        im.nocc = list(nocc)
        return im

    def _init_orbitals(self):
        '''Starting orbitals, occupied first, as lists over spin channels.'''
        mf = self._scf
        mo_energy = self._unpack(numpy.array(mf.mo_energy, copy=True))
        mo_coeff = self._unpack(numpy.array(self.mo_coeff, copy=True))
        mo_occ = self._unpack(numpy.asarray(self.mo_occ))
        out = [occupation.sort_by_occupation(e, c, o)
               for e, c, o in zip(mo_energy, mo_coeff, mo_occ)]
        return [x[0] for x in out], [x[1] for x in out], [x[2] for x in out]

    def kernel(self):
        '''Run the self-consistent calculation and return the total energy.'''
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()
        if self.occupation not in occupation.SCHEMES:
            raise ValueError('occupation must be one of %s' % (occupation.SCHEMES,))
        mo_energy, mo_coeff, nocc = self._init_orbitals()
        self.converged, self.e_tot, mo_energy, mo_coeff, im = scf_loop.kernel(
            self, mo_energy, mo_coeff, nocc)
        self.e_ref = im.e_ref
        if self.e_hf is not None:
            self.e_corr = self.e_tot - self.e_hf
        self.mo_energy = self._pack(mo_energy)
        self.mo_coeff = self._pack(mo_coeff)
        self.mo_occ = self._pack([numpy.where(numpy.arange(c.shape[1]) < n, self._occ, 0.)
                                  for c, n in zip(mo_coeff, nocc)])
        im.blocks = None  # amplitudes are rebuilt on demand
        self._last = im
        self._finalize()
        return self.e_tot

    def _finalize(self):
        name = self.__class__.__name__
        if self.converged:
            logger.note(self, '%s converged.  E(%s) = %.15g', name, name, self.e_tot)
        else:
            logger.warn(self, '%s not converged after %d cycles.  E = %.15g',
                        name, self.max_cycle, self.e_tot)
        return self

    # ---- state after convergence
    def _channels(self):
        '''Converged (mo_energy, mo_coeff, nocc) as lists over channels.'''
        if self.mo_energy is None:
            raise RuntimeError('Run kernel() first.')
        mo_occ = self._unpack(numpy.asarray(self.mo_occ))
        nocc = [int(numpy.count_nonzero(o > 0)) for o in mo_occ]
        return (self._unpack(numpy.asarray(self.mo_energy)),
                self._unpack(numpy.asarray(self.mo_coeff)), nocc)

    def get_intermediates(self, with_amplitudes=False):
        '''Intermediates of the converged solution (rebuilt if amplitudes
        are requested).'''
        if with_amplitudes:
            return self.build(*self._channels())
        if self._last is None:
            raise RuntimeError('Run kernel() first.')
        return self._last

    def get_fock(self, ao_repr=False):
        '''Converged correlated Fock matrix.'''
        im = self.get_intermediates()
        return self._pack(im.fock_ao if ao_repr else im.fock_mo)

    def get_vcorr(self):
        '''Correlation potential in the basis of the converged orbitals.'''
        return self._pack(self.get_intermediates().v)

    def make_rdm1(self, ao_repr=True, correlated=False):
        '''One-particle density matrix.

        Args:
            correlated : False for the density of the OBMP2 determinant
                ("HF-like"); True to add the unrelaxed second-order
                correction built from the converged amplitudes ("MP2-like").
        '''
        return properties.rdm.make_rdm1(self, ao_repr, correlated)

    def dip_moment(self, correlated=False, unit='Debye', verbose=None):
        return properties.rdm.dip_moment(self, correlated, unit, verbose)

    def ip_koopmans(self):
        '''Ionization potential(s) from the highest occupied orbital energy.'''
        return self._pack([-e[n - 1] for e, _, n in zip(*self._channels())])

    def ea_koopmans(self):
        '''Electron affinity(ies) from the lowest virtual orbital energy.'''
        return self._pack([-e[n] for e, _, n in zip(*self._channels())])


class ROBMP2(OBMP2Base):
    '''OBMP2 with a closed-shell restricted reference.'''
    _occ = 2

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if isinstance(mf, scf.uhf.UHF) or isinstance(mf, scf.rohf.ROHF):
            raise TypeError('ROBMP2 needs a closed-shell RHF/RKS reference; '
                            'use UOBMP2 for open shells.')
        OBMP2Base.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def _pack(self, chan):
        return chan[0]

    def _unpack(self, x):
        return [x]

    def spin_square(self):
        return 0., 1.


class UOBMP2(OBMP2Base):
    '''OBMP2 with an unrestricted reference.'''
    _occ = 1

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if not isinstance(mf, scf.uhf.UHF):
            mf = mf.to_uhf()
        OBMP2Base.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def _pack(self, chan):
        if isinstance(chan[0], numpy.ndarray):
            return numpy.array(chan)
        return tuple(chan)

    def _unpack(self, x):
        return [x[0], x[1]]

    def spin_square(self, correlated=False):
        '''<S^2> and 2S+1.

        Args:
            correlated : False for the OBMP2 determinant; True to add the
                first-order amplitude correction (projected value).
        '''
        return properties.spin.spin_square(self, correlated)

    def hfcc(self, atoms=None, correlated=False):
        '''Isotropic hyperfine coupling constants (MHz).'''
        return properties.hfcc.isotropic(self, atoms, correlated)


def OBMP2(mf, frozen=None, mo_coeff=None, mo_occ=None, **kwargs):
    '''OBMP2 for a restricted closed-shell or an unrestricted reference.

    Keyword arguments are set as attributes of the returned object.
    '''
    if isinstance(mf, scf.uhf.UHF) or isinstance(mf, scf.rohf.ROHF):
        mp = UOBMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        mp = ROBMP2(mf, frozen, mo_coeff, mo_occ)
    return mp.set(**kwargs)


def O2BMP2(mf, c_os=None, frozen=None, mo_coeff=None, mo_occ=None, **kwargs):
    '''Spin-opposite-scaled OBMP2: only opposite-spin amplitudes, scaled by
    ``c_os``.

    There is no universal value of ``c_os``; it has to be given explicitly.
    Published choices are 1.2 (singlet-triplet gaps, charge-transfer
    excitations), 1.1 (ionization potentials) and 1.7 (inverted
    singlet-triplet gaps).
    '''
    if c_os is None:
        raise ValueError('O2BMP2 needs an explicit c_os (e.g. c_os=1.2).')
    return OBMP2(mf, frozen, mo_coeff, mo_occ, c_os=c_os, c_ss=0., **kwargs)
