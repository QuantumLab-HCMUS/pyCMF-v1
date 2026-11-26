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
OBMP2 and O2BMP2 with k-point sampling, closed-shell reference.  EXPERIMENTAL.

    >>> from pyscf.pbc import gto, scf
    >>> from pycmf.pbc import KOBMP2
    >>> cell = gto.M(atom='C 0 0 0; C 0.89 0.89 0.89', a=..., basis='gth-dzvp', pseudo='gth-pade')
    >>> mf = scf.KRHF(cell, cell.make_kpts([4, 4, 4])).density_fit().run()
    >>> mp = KOBMP2(mf).run()
    >>> mp.e_tot, mp.ip_koopmans(), mp.ea_koopmans()

The implementation follows the molecular one (:mod:`pycmf.mol.obmp2`): the
self-consistent loop, DIIS and the convergence criteria are those of
:mod:`pycmf.core.scf_loop`, with the irreducible k-points as channels; the
correlation potential is evaluated by :mod:`pycmf.pbc.kbch` from three-index
density-fitted integrals (:mod:`pycmf.pbc.kintegrals`).

The module is under development: it is validated against the previous
four-index implementation and against PySCF's KMP2 (first cycle), but not yet
against published data.
'''

import warnings

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.lib import kpts as libkpts
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.scf import khf

from pycmf.core import mpi, occupation, scf_loop, status
from pycmf.mol.obmp2 import Intermediates
from pycmf.pbc import kbch, kintegrals
from pycmf.pbc.ksymm import KSymmetry


def _as_khf(mf):
    '''Closed-shell k-point Hartree-Fock object on the full k-mesh.'''
    if isinstance(getattr(mf, 'kpts', None), libkpts.KPoints):
        mf = mf.to_khf()
    if not isinstance(mf, khf.KRHF) or hasattr(mf, 'xc'):
        raise NotImplementedError('KOBMP2 needs a closed-shell k-point Hartree-Fock '
                                  'reference (KRHF); got %s' % mf.__class__.__name__)
    return mf


class KOBMP2(lib.StreamObject):
    '''Self-consistent one-body MP2 for periodic systems, closed shell.

    Args:
        mf : converged KRHF object
        frozen : int or list of int
            Orbitals excluded from the amplitudes, at every k-point: the
            lowest ``frozen`` orbitals, or the orbitals with the given
            (energy-ordered) indices, occupied or virtual.  Frozen orbitals
            are still relaxed by the correlated Fock operator.  The frozen set
            must not cut a degenerate level (a warning is issued).

    Attributes:
        conv_tol, conv_tol_grad, max_cycle, diis, diis_space, diis_start_cycle :
            as for molecules (:class:`pycmf.mol.obmp2.OBMP2Base`)
        bch_order : 2 (default) or 1
        c_os, c_ss : scaling of opposite-spin and same-spin amplitudes
        level_shift : shift of the energy denominators
        kpt_symmetry : True (space group and time reversal, default),
            'time_reversal', or False
        with_df : Gaussian density fitting object for the correlation part;
            by default the one of the mean field if it is GDF or RSGDF,
            otherwise a new GDF object

    Saved results:
        converged, e_tot, e_corr (e_tot - E(KRHF)), e_ref (HF energy of the
        OBMP2 determinant), mo_energy, mo_coeff, mo_occ (all k-points of the
        mesh, occupied orbitals first)

    With mpi4py and several MPI processes the loop over k-points of the
    correlation potential is distributed (same script on every rank).
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
    kpt_symmetry = True
    #: store the three-index (ov) tensors in single precision
    single_precision = False
    #: take the exchange matrices of the Hartree-Fock part from the pass over the
    #: three-index integrals that builds the amplitudes (plain KRHF with GDF, no MPI)
    direct_exchange = True

    _keys = {'conv_tol', 'conv_tol_grad', 'max_cycle', 'bch_order', 'c_os', 'c_ss',
             'level_shift', 'occupation', 'diis', 'diis_space', 'diis_start_cycle',
             'kpt_symmetry', 'cell', 'mol', 'kpts', 'nkpts', 'max_memory', 'frozen',
             'with_df', 'mo_energy', 'mo_coeff', 'mo_occ', 'converged', 'e_tot',
             'e_corr', 'e_ref', 'e_hf', 'fock_hf', 'ip', 'ea', 'single_precision',
             'direct_exchange'}

    def __init__(self, mf, frozen=None, kpt_symmetry=None):
        status.experimental(self.__class__.__name__)
        mf = _as_khf(mf)
        self._scf = mf
        self.cell = self.mol = mf.cell
        self.kpts = numpy.asarray(mf.kpts).reshape(-1, 3)
        self.nkpts = len(self.kpts)
        self.verbose = self.cell.verbose
        self.stdout = self.cell.stdout
        self.max_memory = mf.max_memory
        self.frozen = frozen
        if kpt_symmetry is not None:
            self.kpt_symmetry = kpt_symmetry
        self.with_df = None

        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.converged = False
        self.e_hf = getattr(mf, 'e_tot', None)
        self.e_tot = None
        self.e_corr = None
        self.e_ref = None
        self.fock_hf = None
        self.ip = None
        self.ea = None
        self._se = None
        self._sym = None
        self._ints = None
        self._cache = {}
        self._last = None

    # ---- setup
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('nkpts = %d', self.nkpts)
        log.info('bch_order = %d', self.bch_order)
        log.info('c_os = %g  c_ss = %g', self.c_os, self.c_ss)
        log.info('level_shift = %g', self.level_shift)
        log.info('frozen = %s', self.frozen)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('DIIS = %s, space = %d', self.diis, self.diis_space)
        log.info('kpt_symmetry = %s', self.kpt_symmetry)
        log.info('MPI processes = %d', mpi.size)
        log.info('max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        return self

    def density_fit(self, auxbasis=None, with_df=None):
        '''Choose the density fitting of the correlation potential: a GDF
        object, or an auxiliary basis for a new one.'''
        self.with_df = kintegrals.as_gdf(self._scf, auxbasis, with_df)
        self._ints = None
        return self

    def get_ovlp(self):
        '''Overlap matrices of the irreducible k-points (channels of the
        self-consistent loop).'''
        s1e = self._overlap()
        return [s1e[k] for k in self._sym.ibz2bz]

    def _overlap(self):
        if 's1e' not in self._cache:
            self._cache['s1e'] = numpy.asarray(self._scf.get_ovlp())
        return self._cache['s1e']

    def _hcore(self):
        if 'h1e' not in self._cache:
            self._cache['h1e'] = numpy.asarray(mpi.root_call(self._scf.get_hcore))
        return self._cache['h1e']

    def _kconserv(self):
        if 'kconserv' not in self._cache:
            self._cache['kconserv'] = kpts_helper.get_kconserv(self.cell, self.kpts)
        return self._cache['kconserv']

    def _init_orbitals(self):
        '''Orbitals of the mean field at all k-points, occupied first.

        Orbitals that PySCF removed because of linear dependencies of the
        Bloch basis (zero vectors) are dropped, so that the number of orbitals
        may depend on the k-point.
        '''
        mf = self._scf
        s1e = self._overlap()
        mo_energy, mo_coeff, nocc = [], [], []
        # restart: orbital energies set by the user together with mo_coeff, mo_occ
        mo_e = mf.mo_energy if self.mo_energy is None else self.mo_energy
        for k, (e, c, o) in enumerate(zip(mo_e, self.mo_coeff, self.mo_occ)):
            e, c, o = numpy.asarray(e), numpy.asarray(c), numpy.asarray(o)
            if abs(o[o > 1e-8] - 2).max(initial=0.) > 1e-8:
                raise NotImplementedError('fractional or single occupations')
            keep = numpy.einsum('pi,pi->i', c.conj(), s1e[k].dot(c)).real > .5
            e, c, n = occupation.sort_by_occupation(e[keep], c[:, keep], o[keep])
            mo_energy.append(e)
            mo_coeff.append(c)
            nocc.append(int(n))
        return mo_energy, mo_coeff, nocc

    def _init_symmetry(self, mo_energy, mo_coeff, nocc, log):
        '''Irreducible k-points; fall back to the full mesh if the mean-field
        solution does not have the symmetry of the crystal.'''
        mode = self.kpt_symmetry
        if mode not in (True, False, 'time_reversal'):
            raise ValueError("kpt_symmetry must be True, False or 'time_reversal'")
        sym = KSymmetry(self.cell, self.kpts, space_group=(mode is True), time_reversal=bool(mode))
        if not sym.trivial:
            mo_bz = sym.expand([mo_coeff[k] for k in sym.ibz2bz])
            err = 0.
            for k, i in enumerate(sym.bz2ibz):
                kr = sym.ibz2bz[i]
                n = nocc[k]
                if n != nocc[kr] or mo_energy[k].size != mo_energy[kr].size:
                    err = numpy.inf
                    break
                err = max(err, abs(mo_energy[k] - mo_energy[kr]).max())
                d0 = mo_coeff[k][:, :n].dot(mo_coeff[k][:, :n].conj().T)
                d1 = mo_bz[k][:, :n].dot(mo_bz[k][:, :n].conj().T)
                err = max(err, abs(d0 - d1).max())
            if err > 1e-4:
                log.warn('The mean-field solution does not have the symmetry of the '
                         'k-point mesh (deviation %.2g); k-point symmetry is not used.', err)
                sym = KSymmetry(self.cell, self.kpts, space_group=False, time_reversal=False)
        log.info('k-points: %d in the mesh, %d irreducible (%d symmetry operations)',
                 sym.nkpts, sym.nibz, sym.nops)
        return sym

    def _active(self, nocc, nmo, mo_energy=None):
        '''Index arrays of the correlated occupied and virtual orbitals.'''
        frozen = self.frozen
        if frozen is None:
            frozen = []
        elif isinstance(frozen, (int, numpy.integer)):
            frozen = range(frozen)
        frozen = set(int(x) for x in frozen)
        occ = [numpy.array([i for i in range(n) if i not in frozen], dtype=int) for n in nocc]
        vir = [numpy.array([a for a in range(n, m) if a not in frozen], dtype=int)
               for n, m in zip(nocc, nmo)]
        if frozen and mo_energy is not None and not self._cache.get('frozen_checked'):
            # a frozen set that cuts a degenerate level is not invariant under
            # rotations of the degenerate orbitals: the energy is then ill defined
            self._cache['frozen_checked'] = True
            for k, e in enumerate(mo_energy):
                for p in range(len(e) - 1):
                    if ((p in frozen) != (p + 1 in frozen)) and abs(e[p + 1] - e[p]) < 1e-6:
                        warnings.warn('KOBMP2: the frozen orbitals cut the degenerate level %d/%d at '
                                      'k-point %d; the energy then depends on the arbitrary rotation '
                                      'of the degenerate orbitals.' % (p, p + 1, k))
                        return occ, vir
        return occ, vir

    # ---- one cycle
    def _hf_part(self, dm, h1e, vk=None):
        '''Hartree-Fock potential and energy of the density matrices ``dm``.
        With ``vk`` (exchange matrices from :meth:`KDFInts.build_rows`) only
        the Coulomb part is computed here.'''
        mf = self._scf
        if vk is None:
            vhf = numpy.asarray(mf.get_veff(self.cell, dm))
        else:
            vj = numpy.asarray(mf.get_j(self.cell, dm, hermi=1))
            if mf.exxdiv == 'ewald':
                from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
                vk = numpy.array(vk)
                _ewald_exxdiv_for_G0(self.cell, self.kpts, dm[None], vk[None])
            vhf = vj - .5 * vk
        e_hf = mf.energy_elec(dm, h1e, vhf)[0].real + mf.energy_nuc()
        return vhf, float(e_hf)

    def _exchange_from_rows(self):
        '''True if the exchange matrices of Hartree-Fock can be taken from the
        pass over the three-index integrals that builds the amplitudes: plain
        KRHF on the same Gaussian density fitting object, no MPI.'''
        mf = self._scf
        return (self.direct_exchange and mpi.size == 1
                and isinstance(mf, khf.KRHF) and not hasattr(mf, 'xc')
                and isinstance(mf.kpts, numpy.ndarray)
                and (self.with_df is None or self.with_df is mf.with_df)
                and kintegrals.as_gdf(mf) is mf.with_df
                and mf.exxdiv in (None, 'ewald')
                and getattr(mf, 'rsjk', None) is None)

    def build(self, mo_energy, mo_coeff, nocc):
        '''Correlated Fock matrices and energy for a given set of orbitals.

        All arguments are lists over the irreducible k-points; orbitals are
        ordered as [occupied | virtual].
        '''
        log = logger.new_logger(self)
        cput0 = (logger.process_clock(), logger.perf_counter())
        sym = self._sym
        nk = sym.nkpts
        if mpi.comm is not None:
            mo_energy, mo_coeff = mpi.bcast((mo_energy, mo_coeff))
        s1e = self._overlap()
        h1e = self._hcore()

        # orbitals of the full zone in a symmetry-consistent gauge
        mo_bz = sym.expand(mo_coeff)
        e_bz = sym.expand_scalar(mo_energy)
        nocc_bz = sym.expand_scalar(nocc)
        nmo_bz = [c.shape[1] for c in mo_bz]

        direct = self._exchange_from_rows()
        occ, vir = self._active(nocc_bz, nmo_bz, e_bz)
        if self._ints is None:
            if self.with_df is None:
                self.with_df = kintegrals.as_gdf(self._scf)
            self._ints = kintegrals.KDFInts(self.with_df, self.kpts, self.max_memory,
                                            self.verbose, self.stdout)
        ints = self._ints
        ints.single_precision = self.single_precision
        nrows = len(set(mpi.split(range(nk))) | set(int(k) for k in sym.ibz2bz))
        ints.update(mo_bz, occ, vir, nrows, nocc_bz)
        # one pass over the integrals: (ov) tensors of all k-points and, if
        # possible, the exchange matrices of the Hartree-Fock part
        vk = ints.build_rows(with_k=True) if direct else None

        dm = numpy.asarray([2 * c[:, :n].dot(c[:, :n].conj().T) for c, n in zip(mo_bz, nocc_bz)])
        vhf, e_hf = mpi.root_call(self._hf_part, dm, h1e, vk)
        fock_hf_ao = h1e + vhf
        fock_hf = [c.conj().T.dot(f).dot(c) for c, f in zip(mo_bz, fock_hf_ao)]
        cput1 = log.timer_debug1('HF part and (ov) integrals' if direct else 'HF part', *cput0)

        v, c0, rdm = kbch.correlation_potential(
            ints, sym, self._kconserv(), e_bz, mo_bz, s1e, fock_hf, occ, vir,
            self.c_os, self.c_ss, self.level_shift, self.bch_order, log)
        log.debug1('  reading AO integrals %.2f s, transformation %.2f s', ints.t_ao, ints.t_mo)
        ints.t_ao = ints.t_mo = 0.
        log.timer_debug1('correlation potential', *cput1)

        e_c = c0
        fock_ao, fock_mo, dm_chan = [], [], []
        for i, k in enumerate(sym.ibz2bz):
            n = nocc[i]
            e_c += 2. * sym.star[i] / nk * numpy.trace(v[i][:n, :n]).real
            sc = s1e[k].dot(mo_bz[k])
            f = fock_hf_ao[k] + sc.dot(v[i]).dot(sc.conj().T)
            c = mo_coeff[i]
            fock_ao.append(f)
            fock_mo.append(c.conj().T.dot(f).dot(c))
            dm_chan.append(c[:, :n].dot(c[:, :n].conj().T))

        im = Intermediates()
        im.e_ref = im.e_hf = e_hf
        im.e_corr = e_c
        im.e_tot = e_hf + e_c
        im.dm_chan = dm_chan
        im.fock_ao = fock_ao
        im.fock_mo = fock_mo
        im.fock_hf = fock_hf
        im.v = v
        im.c0 = c0
        im.rdm = rdm
        im.nocc = list(nocc)
        return im

    # ---- driver
    def kernel(self):
        '''Run the self-consistent calculation and return the total energy
        per unit cell.'''
        if mpi.rank > 0:
            self.verbose = 0
        log = logger.new_logger(self)
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()
        if self.occupation != 'aufbau':
            raise NotImplementedError("only occupation='aufbau' for periodic systems")
        mo_energy, mo_coeff, nocc = mpi.bcast(self._init_orbitals())
        sym = self._sym = self._init_symmetry(mo_energy, mo_coeff, nocc, log)
        self._ints = None
        ibz = sym.ibz2bz
        nocc_ibz = [nocc[k] for k in ibz]
        self.converged, self.e_tot, mo_energy, mo_coeff, im = scf_loop.kernel(
            self, [mo_energy[k] for k in ibz], [mo_coeff[k] for k in ibz], nocc_ibz)

        self.mo_energy = sym.expand_scalar(mo_energy)
        self.mo_coeff = sym.expand(mo_coeff)
        self.mo_occ = []
        for e, n in zip(self.mo_energy, sym.expand_scalar(nocc_ibz)):
            o = numpy.zeros(e.size)
            o[:n] = 2
            self.mo_occ.append(o)
        self.e_ref = im.e_ref
        self.e_corr = self.e_tot - self.e_hf if self.e_hf is not None else None
        # HF Fock matrices in the basis of the final orbitals (all k-points)
        self.fock_hf = im.fock_hf
        self._last = im
        self._ints = None
        self._finalize()
        return self.e_tot

    def _finalize(self):
        log = logger.new_logger(self)
        if self.converged:
            log.note('%s converged', self.__class__.__name__)
        else:
            log.warn('%s not converged in %d cycles', self.__class__.__name__, self.max_cycle)
        log.note('E(%s) = %.15g  E_corr = %.15g', self.__class__.__name__, self.e_tot, self.e_corr)
        return self

    # ---- results
    def get_fock(self):
        '''Correlated Fock matrices (AO basis) at all k-points of the mesh.'''
        im, sym = self._last, self._sym
        if sym.trivial:
            return numpy.asarray(im.fock_ao)
        return numpy.asarray(sym.kd.transform_fock(numpy.asarray(im.fock_ao)))

    def ip_koopmans(self):
        '''Ionization potential: minus the valence band maximum on the mesh.'''
        return -max(e[o > 0].max() for e, o in zip(self.mo_energy, self.mo_occ))

    def ea_koopmans(self):
        '''Electron affinity: minus the conduction band minimum on the mesh.'''
        return -min(e[o == 0].min() for e, o in zip(self.mo_energy, self.mo_occ))

    def band_gap(self):
        '''Gap between the eigenvalues of the correlated Fock matrix on the
        k-mesh.  These eigenvalues are not removal and addition energies; see
        :meth:`ipea` for the quasiparticle gap.'''
        return self.ip_koopmans() - self.ea_koopmans()

    def init_from_scf(self):
        '''Take the orbitals and energies of the mean field without running
        OBMP2 (for conventional second-order quasiparticle energies).'''
        log = logger.new_logger(self)
        mo_energy, mo_coeff, nocc = mpi.bcast(self._init_orbitals())
        sym = self._sym = self._init_symmetry(mo_energy, mo_coeff, nocc, log)
        ibz = sym.ibz2bz
        self.mo_energy = sym.expand_scalar([mo_energy[k] for k in ibz])
        self.mo_coeff = sym.expand([mo_coeff[k] for k in ibz])
        self.mo_occ = []
        for e, n in zip(self.mo_energy, sym.expand_scalar([nocc[k] for k in ibz])):
            o = numpy.zeros(e.size)
            o[:n] = 2
            self.mo_occ.append(o)
        return self

    def self_energy(self, nvir=2, kpts_index=None, grid=None, mo_energy=None, nocc=None,
                    scaling='linear'):
        '''Tabulated second-order self-energy of the band-edge states, see
        :mod:`pycmf.pbc.kqp`.  The result is kept in ``self._se``.'''
        from pycmf.pbc import kqp
        self._se = kqp.self_energy(self, nvir, kpts_index, grid, mo_energy, nocc, scaling)
        return self._se

    def quasiparticle_gap(self, scheme='static', reference='obmp2'):
        '''(direct gap at the first k-point, smallest gap on the mesh) from the
        quasiparticle energies of :func:`pycmf.pbc.kqp.quasiparticle`; needs
        :meth:`self_energy`.'''
        from pycmf.pbc import kqp
        e_qp, _ = kqp.quasiparticle(self._se, scheme, reference)
        return kqp.gaps(self._se, e_qp)[:2]

    def peom_mp2(self, kpt_v=0, kpts_c=None, symmetry=True):
        '''Band edges from partitioned EOM-MP2 on the present orbitals, see
        :mod:`pycmf.pbc.kpeom`.  Returns a dict with the quasiparticle energy
        of the highest occupied band at k-point ``kpt_v`` ('ip'), those of the
        lowest virtual band at the k-points ``kpts_c`` ('ea') and the gaps
        ('gap').'''
        from pycmf.pbc import kpeom
        return kpeom.gap(self, kpt_v, kpts_c, symmetry)

    def downfold(self, ncas, nelecas, cas_list=None):
        '''Active-space Hamiltonian downfolded with the present correlation
        potential, see :class:`pycmf.pbc.kobdf.KOBDF`.'''
        from pycmf.pbc.kobdf import KOBDF
        return KOBDF(self, ncas, nelecas, cas_list)

    def ipea(self, nvir=4, kpts_index=None):
        '''Second-order IP/EA corrections (IPEA-KOBMP2).

        Sets ``self.ip`` and ``self.ea`` (lists over the k-points; all
        correlated occupied bands and the ``nvir`` lowest virtual bands) and
        returns the quasiparticle gap  min(IP) - max(EA)  on the mesh.
        '''
        from pycmf.pbc import kipea, kqp
        self.ip, self.ea = kipea.kernel(self, nvir, kpts_index)
        se = [x for x in self._se if x is not None]
        e_qp, _ = kqp.quasiparticle(se, 'static', 'obmp2')
        return kqp.gaps(se, e_qp)[1]

    # ---- names of the previous implementation
    niter = property(lambda self: self.max_cycle,
                     lambda self, x: setattr(self, 'max_cycle', x))
    thresh = property(lambda self: self.conv_tol,
                      lambda self, x: setattr(self, 'conv_tol', x))
    shift = property(lambda self: self.level_shift,
                     lambda self, x: setattr(self, 'level_shift', x))
    ene_tot = property(lambda self: self.e_tot)


def remove_linear_dependency_(mf, threshold=1e-4):
    '''Let a k-point mean-field object work in the space of the eigenvectors
    of the overlap matrix with eigenvalues above ``threshold`` (canonical
    orthogonalization, per k-point).  Call it before ``mf.kernel()``.

    PySCF removes only the functions that make the overlap matrix numerically
    singular.  The self-consistent iterations of OBMP2 are less robust than
    those of Hartree-Fock: with nearly dependent functions left in the basis
    (diamond with gth-tzvp: overlap eigenvalues down to 1e-6) they do not
    converge, and they converge in the usual five to seven cycles once those
    functions are removed, at a cost of 5e-5 Hartree in the Hartree-Fock
    energy.  KOBMP2 keeps the orbital space of the mean field.
    '''
    import scipy.linalg

    def eig(h_kpts, s_kpts, x=None):
        nkpts, nao = len(h_kpts), h_kpts[0].shape[0]
        mo_energy = numpy.full((nkpts, nao), 1e30)
        mo_coeff = numpy.zeros((nkpts, nao, nao), dtype=numpy.complex128)
        for k in range(nkpts):
            w, u = scipy.linalg.eigh(s_kpts[k])
            keep = w > threshold
            x_k = u[:, keep] / numpy.sqrt(w[keep])
            e, c = scipy.linalg.eigh(x_k.conj().T.dot(h_kpts[k]).dot(x_k))
            mo_energy[k, :e.size] = e
            mo_coeff[k, :, :e.size] = x_k.dot(c)
        return mo_energy, mo_coeff

    mf.eig = eig
    return mf


def KO2BMP2(mf, c_os=None, **kwargs):
    '''k-point OBMP2 with scaled opposite-spin amplitudes only.

    ``c_os`` has no default: it is an empirical parameter that depends on the
    property of interest.
    '''
    if c_os is None:
        raise ValueError('KO2BMP2 needs an explicit c_os.')
    mp = KOBMP2(mf, **kwargs)
    mp.c_os = c_os
    mp.c_ss = 0.
    return mp
