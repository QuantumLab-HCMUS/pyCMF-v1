'''k-point OBMP2 (experimental).

References:
  * the previous four-index implementation (pycmf 0.2 / commit 1e9b7c1) with
    Gaussian density fitting, cycle by cycle without DIIS;
  * PySCF's KMP2 for the first cycle;
  * the molecular implementation, fed with the integrals of a Gamma-point cell.
'''

import os
import shutil
import subprocess
import sys

import numpy
import pytest
from pyscf import ao2mo
from pyscf import gto as mgto
from pyscf import scf as mscf
from pyscf.pbc import gto, scf
from pyscf.pbc import mp as pmp

import pycmf
from pycmf.core.status import ExperimentalWarning
from pycmf.pbc import KOBMP2, KO2BMP2

pytestmark = pytest.mark.filterwarnings('ignore::pycmf.core.status.ExperimentalWarning')

A0 = 3.5668


def diamond(dx=0.):
    return gto.M(atom='C 0 0 0; C %.10f %.10f %.10f' % (A0 / 4 + dx, A0 / 4, A0 / 4 - 0.5 * dx),
                 a=[[0, A0 / 2, A0 / 2], [A0 / 2, 0, A0 / 2], [A0 / 2, A0 / 2, 0]],
                 basis='gth-szv', pseudo='gth-pade', verbose=0)


def boron_nitride():
    a = 3.615
    return gto.M(atom='B 0 0 0; N %.10f %.10f %.10f' % (a / 4, a / 4, a / 4),
                 a=[[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]],
                 basis='gth-szv', pseudo='gth-pade', verbose=0)


def krhf(cell, mesh, **kwargs):
    mf = scf.KRHF(cell, cell.make_kpts(mesh), **kwargs).density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    return mf


def run(mf, cls=KOBMP2, max_cycle=None, **kwargs):
    keys = ('frozen', 'kpt_symmetry') + (('c_os',) if cls is KO2BMP2 else ())
    init = {k: kwargs.pop(k) for k in keys if k in kwargs}
    mp = cls(mf, **init)
    mp.conv_tol = 1e-10
    mp.__dict__.update(kwargs)
    if max_cycle is not None:
        mp.max_cycle = max_cycle
    mp.kernel()
    return mp


@pytest.fixture(scope='module')
def c112():
    return krhf(diamond(), [1, 1, 2])


@pytest.fixture(scope='module')
def c222():
    return krhf(diamond(), [2, 2, 2])


# Energies of the previous implementation (diamond, gth-szv, GDF) after 1 and
# 3 cycles without DIIS, and at convergence.
OLD = {
    'c112': (-10.642769835772, -10.642595233639, -10.642599010455),
    'c222': (-11.026958465375, -11.026880327237, -11.026881350304),
}


def test_is_experimental(c112):
    assert not hasattr(pycmf, 'KOBMP2')
    with pytest.warns(ExperimentalWarning):
        KOBMP2(c112)


def test_previous_implementation(c112):
    e1, e3, e_conv = OLD['c112']
    assert abs(run(c112, max_cycle=1, diis=False).e_tot - e1) < 1e-9
    assert abs(run(c112, max_cycle=3, diis=False).e_tot - e3) < 1e-9
    mp = run(c112)
    assert mp.converged
    assert abs(mp.e_tot - e_conv) < 1e-8
    # names of the previous implementation
    assert mp.ene_tot == mp.e_tot
    assert mp.thresh == mp.conv_tol and mp.niter == mp.max_cycle
    assert mp.band_gap() > 0


def test_first_cycle_is_kmp2(c112):
    pt = pmp.KMP2(c112)
    pt.kernel()
    assert abs(run(c112, max_cycle=1).e_tot - (c112.e_tot + pt.e_corr)) < 1e-9
    pt = pmp.KMP2(c112, frozen=1)
    pt.kernel()
    assert abs(run(c112, max_cycle=1, frozen=1).e_tot - (c112.e_tot + pt.e_corr)) < 1e-9
    # scaled amplitudes: the first-cycle energy is the Hylleraas functional
    pt = pmp.KMP2(c112)
    pt.kernel()
    if hasattr(pt, 'e_corr_os'):
        h = lambda c: 2 * c - c * c
        e = run(c112, max_cycle=1, c_os=1.2, c_ss=1. / 3).e_tot
        assert abs(e - c112.e_tot - h(1.2) * pt.e_corr_os - h(1. / 3) * pt.e_corr_ss) < 1e-9


@pytest.mark.slow
def test_frozen_cutting_a_degenerate_level_warns(c222):
    '''In diamond the two lowest valence bands are degenerate at the X point.'''
    with pytest.warns(UserWarning, match='degenerate'):
        run(c222, max_cycle=1, frozen=1)


def test_ipea(c112):
    '''IP/EA corrections against a direct evaluation with four-index integrals.'''
    mp = run(c112)
    gap = mp.ipea(nvir=1)
    assert gap < mp.band_gap()
    kpts, nk, no = mp.kpts, 2, 4
    kconserv, c, e = mp._kconserv(), mp.mo_coeff, mp.mo_energy
    ao2mo = c112.with_df.ao2mo

    def eri(ks, cs):
        shape = [x.shape[1] for x in cs]
        return ao2mo(cs, kpts[list(ks)], compact=False).reshape(shape) / nk

    h, d_ip, d_ea = no - 1, 0., 0.
    for ki in range(nk):
        for ka in range(nk):
            ci, ca = c[ki][:, :no], c[ka][:, no:]
            e_ia = e[ki][:no, None] - e[ka][None, no:]
            kj = kconserv[ka, ki, 0]
            g = eri((ki, ka, kj, 0), (ci, ca, c[kj][:, :no], c[0][:, h:h + 1]))
            gx = eri((kj, ka, ki, 0), (c[kj][:, :no], ca, ci, c[0][:, h:h + 1])).transpose(2, 1, 0, 3)
            den = e_ia[:, :, None, None] + (e[kj][:no] - e[0][h])[None, None, :, None]
            d_ip += 2 * ((g - .5 * gx).conj() / den * g).sum().real
            kb = kconserv[ki, ka, 0]
            g = eri((ki, ka, 0, kb), (ci, ca, c[0][:, no:no + 1], c[kb][:, no:]))
            gx = eri((ki, kb, 0, ka), (ci, c[kb][:, no:], c[0][:, no:no + 1], ca)).transpose(0, 3, 2, 1)
            den = e_ia[:, :, None, None] + (e[0][no] - e[kb][no:])[None, None, None, :]
            d_ea -= 2 * ((g - .5 * gx).conj() / den * g).sum().real
    assert abs(mp.ip[0][-1] - (-e[0][h] + d_ip)) < 1e-8
    assert abs(mp.ea[0][0] - (-e[0][no] + d_ea)) < 1e-8
    assert d_ip < 0 and d_ea > 0


def test_self_energy_halves(c112):
    '''With HF orbitals the diagonal of the OBMP2 potential is one half of the
    static second-order self-energy: 2p1h for occupied, 2h1p for virtual bands.'''
    from pycmf.pbc import kqp
    mp = run(c112, max_cycle=1)
    se = mp.self_energy(nvir=4)
    for i, k in enumerate(mp._sym.ibz2bz):
        v = numpy.diag(mp._last.v[i]).real
        assert abs(v[:4] - se[k]['A0'][:4]).max() < 1e-8
        assert abs(v[4:] - se[k]['B0'][4:]).max() < 1e-8
        assert (se[k]['A0'][:4] < 0).all() and (se[k]['B0'][4:] > 0).all()
    # conventional second-order quasiparticle gaps with the HF propagator
    mp = KOBMP2(c112).init_from_scf()
    se = mp.self_energy(nvir=2)
    gap = {}
    for scheme in ('static', 'linear', 'dynamic'):
        e, z = kqp.quasiparticle(se, scheme, 'hf')
        gap[scheme] = kqp.gaps(se, e)[1]
        assert all(((x > 0.5) & (x < 1)).all() for x in z)
    hf = min(e[4] for e in c112.mo_energy) - max(e[3] for e in c112.mo_energy)
    assert gap['static'] < gap['dynamic'] < hf
    assert abs(gap['linear'] - gap['dynamic']) < 2e-3


def test_self_energy_scaling_and_bands(c112):
    '''Spin-component scaled variants: the potential carries the self-energy
    with the factors 2c - c^2 (Hylleraas form), not c.'''
    mp = run(c112, max_cycle=1, c_os=1.2, c_ss=1. / 3)
    hyl = mp.self_energy(nvir=4, grid=[0.], scaling='hylleraas')
    lin = mp.self_energy(nvir=4, grid=[0.])
    dev_hyl = dev_lin = 0.
    for i, k in enumerate(mp._sym.ibz2bz):
        v = numpy.diag(mp._last.v[i]).real
        dev_hyl = max(dev_hyl, abs(v[:4] - hyl[k]['A0'][:4]).max(), abs(v[4:] - hyl[k]['B0'][4:]).max())
        dev_lin = max(dev_lin, abs(v[:4] - lin[k]['A0'][:4]).max(), abs(v[4:] - lin[k]['B0'][4:]).max())
    assert dev_hyl < 1e-8
    assert dev_lin > 1e-4
    # without same-spin part the two differ by the factor (2c - c^2) / c
    mp = run(c112, max_cycle=1, c_os=1.3, c_ss=0.)
    hyl = mp.self_energy(nvir=2, scaling='hylleraas')
    lin = mp.self_energy(nvir=2)
    assert abs(hyl[0]['B'] - .7 * lin[0]['B']).max() < 1e-10
    # selection of bands
    few = mp.self_energy(nvir=1, nocc=1)
    assert few[0]['e'].size == 2 and few[0]['occ'].tolist() == [True, False]
    assert abs(few[0]['A'] - lin[0]['A'][[3, 4]]).max() < 1e-12
    assert abs(few[0]['dB'] - lin[0]['dB'][[3, 4]]).max() < 1e-12


def test_quasiparticle_schemes(c112):
    mp = run(c112)
    static = mp.ipea(nvir=2)
    mp.self_energy(nvir=2)
    d0, g0 = mp.quasiparticle_gap('static')
    d1, g1 = mp.quasiparticle_gap('linear')
    d2, g2 = mp.quasiparticle_gap('dynamic')
    assert abs(g0 - static) < 1e-10
    assert g0 < g2 < mp.band_gap() and abs(g1 - g2) < 3e-3
    # single-precision storage of the three-index tensors
    sp = run(c112, single_precision=True)
    assert abs(sp.e_tot - mp.e_tot) < 1e-7


def test_exchange_from_integral_pass(c112):
    '''Hartree-Fock exchange accumulated in the pass that builds the (ov)
    tensors against the exchange matrices of PySCF (exxdiv='ewald').'''
    ref = run(c112, direct_exchange=False)
    new = run(c112)
    assert new._exchange_from_rows() and not ref._exchange_from_rows()
    assert abs(new.e_tot - ref.e_tot) < 1e-10
    assert max(abs(a - b).max() for a, b in zip(new.mo_energy, ref.mo_energy)) < 1e-8
    mf = krhf(diamond(), [1, 1, 2], exxdiv=None)
    assert abs(run(mf).e_tot - run(mf, direct_exchange=False).e_tot) < 1e-10


def _pyscf_peom(mf, mo_coeff, mo_energy, kpt):
    '''P-EOM-MP2 with the EOM-KCCSD code of PySCF: t1 = 0, MP2-like t2 with
    the given orbital energies, which also fill the doubles-doubles block.'''
    from pyscf.pbc import cc
    from pyscf.pbc.cc import eom_kccsd_rhf
    mycc = cc.KRCCSD(mf, mo_coeff=numpy.asarray(mo_coeff))
    mycc.verbose = 0
    mycc.keep_exxdiv = True
    eris = mycc.ao2mo()
    eris.mo_energy = [numpy.array(e) for e in mo_energy]
    _, t1, t2 = mycc.init_amps(eris)
    mycc.t1, mycc.t2, mycc.eris = numpy.zeros_like(t1), t2, eris
    imds = eom_kccsd_rhf._IMDS(mycc, eris=eris)
    imds.make_ip('mp')
    imds.make_ea('mp')
    nocc = mycc.nocc
    fock = eris.fock.copy()
    for k, e in enumerate(mo_energy):
        fock[k, :nocc, :nocc] = numpy.diag(e[:nocc])
        fock[k, nocc:, nocc:] = numpy.diag(e[nocc:])
    eris.fock = fock
    out = []
    for cls in (eom_kccsd_rhf.EOMIP, eom_kccsd_rhf.EOMEA):
        eom = cls(mycc)
        eom.partition, eom.verbose, eom.eris, eom.conv_tol = 'mp', 0, eris, 1e-9
        e, _ = eom.kernel(nroots=1, koopmans=True, imds=imds, kptlist=[kpt])
        out.append(float(numpy.ravel(e)[0]))
    return out


@pytest.mark.parametrize('reference', ['hf', 'obmp2'])
def test_peom_mp2_against_pyscf(reference):
    '''Complex orbitals, no symmetry; with OBMP2 orbitals the Fock matrix of
    Hartree-Fock is not diagonal, which tests the f_ov terms.'''
    from pycmf.pbc import kpeom
    cell = diamond(0.15)
    mf = scf.KRHF(cell, cell.make_kpts([1, 1, 2], scaled_center=[0.11, 0.23, 0.07]), exxdiv=None).density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    mp = KOBMP2(mf, kpt_symmetry=False)
    if reference == 'hf':
        mp.init_from_scf()
    else:
        mp.conv_tol = 1e-10
        mp.kernel()
    setup = kpeom._Setup(mp)
    for kpt in (0, 1):
        ip_ref, ea_ref = _pyscf_peom(mf, mp.mo_coeff, mp.mo_energy, kpt)
        fine = numpy.linspace(-.02, .02, 9)
        r = kpeom.ip(mp, kpt, symmetry=False, setup=setup)
        r = kpeom.ip(mp, kpt, symmetry=False, setup=setup, grid=fine + (r['e_ref'] - r['e_qp']))
        assert abs(-r['e_qp'] - ip_ref) < 1e-7
        r = kpeom.ea(mp, kpt, symmetry=False, setup=setup)
        r = kpeom.ea(mp, kpt, symmetry=False, setup=setup, grid=fine + (r['e_qp'] - r['e_ref']))
        assert abs(r['e_qp'] - ea_ref) < 1e-7


def test_peom_mp2_symmetry(c222):
    '''Sum over the k-points inequivalent under the little group and trace
    over the degenerate level against the full sum.'''
    from pycmf.pbc import kpeom
    mp = KOBMP2(c222).init_from_scf()
    setup = kpeom._Setup(mp)
    kc = int(numpy.argmin([e[0] for e in setup.ev]))
    assert len(kpeom._wedge(mp._sym, 0, True)[0]) == 3
    r = kpeom.ip(mp, 0, setup=setup)
    full = kpeom.ip(mp, 0, symmetry=False, setup=setup)
    assert abs(r['e_qp'] - full['e_qp']) < 1e-8
    for kpt in {0, kc}:
        level = kpeom._level(setup.ev[kpt], 0)
        r = kpeom.ea(mp, kpt, setup=setup)
        full = kpeom.ea(mp, kpt, nbands=level, symmetry=False, setup=setup)
        assert abs(r['e_qp'] - full['e_qp']) < 1e-8
    res = kpeom.gap(mp, kpts_c=[0])
    assert 0.1 < res['gap'][0] < c222.mo_energy[0][4] - c222.mo_energy[0][3]


def test_remove_linear_dependency():
    '''A mean field in a reduced orbital space: KOBMP2 keeps that space.'''
    from pycmf.pbc import remove_linear_dependency_
    cell = diamond()
    mf = scf.KRHF(cell, cell.make_kpts([1, 1, 2])).density_fit()
    s = mf.get_ovlp()
    cut = sorted(numpy.concatenate([numpy.linalg.eigvalsh(x) for x in s]))[1]
    remove_linear_dependency_(mf, threshold=cut * 1.0001)
    mf.conv_tol = 1e-11
    mf.kernel()
    mp = run(mf)
    assert mp.converged
    assert sum(len(e) for e in mp.mo_energy) == 2 * cell.nao - 2
    assert abs(mp.e_tot - mf.e_tot) > 1e-3


def test_ko2bmp2(c112):
    with pytest.raises(ValueError):
        KO2BMP2(c112)
    a = run(c112, cls=KO2BMP2, c_os=1.3)
    b = run(c112, c_os=1.3, c_ss=0.)
    assert a.converged and abs(a.e_tot - b.e_tot) < 1e-12


def test_rows_on_disk(c112):
    a = run(c112, max_cycle=2)
    b = run(c112, max_cycle=2, max_memory=0.01)
    assert abs(a.e_tot - b.e_tot) < 1e-11


@pytest.mark.parametrize('opts', [{}, {'c_os': 1.3, 'c_ss': 0.}, {'c_os': 1.2, 'c_ss': 0.4},
                                  {'level_shift': 0.2}, {'bch_order': 1}])
def test_gamma_point_equals_molecule(opts, gamma_pair):
    '''One k-point: same result as the molecular code with the same integrals.'''
    kmf, mf = gamma_pair
    a = pycmf.OBMP2(mf)
    a.__dict__.update(opts)
    a.conv_tol = 1e-11
    a.kernel()
    b = run(kmf, conv_tol=1e-11, **opts)
    assert a.converged and b.converged
    assert abs(a.e_tot - b.e_tot) < 2e-8


@pytest.fixture(scope='module')
def gamma_pair():
    cell = boron_nitride()
    kmf = krhf(cell, [1, 1, 1], exxdiv=None)
    nao = cell.nao
    h1 = kmf.get_hcore()[0].real
    s1 = kmf.get_ovlp()[0].real
    eri = numpy.asarray(kmf.with_df.get_eri(compact=False)).real.reshape([nao] * 4)
    mol = mgto.M(verbose=0)
    mol.nelectron = cell.nelectron
    mol.incore_anyway = True
    mf = mscf.RHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: s1
    mf.energy_nuc = lambda *args: kmf.energy_nuc()
    mf._eri = ao2mo.restore(8, eri, nao)
    mf.conv_tol = 1e-12
    mf.kernel(dm0=kmf.make_rdm1()[0].real)
    assert abs(mf.e_tot - kmf.e_tot) < 1e-9
    return kmf, mf


# ---- slower tests: symmetry, larger meshes, MPI

@pytest.mark.slow
@pytest.mark.parametrize('mode', [False, True])
def test_previous_implementation_222(c222, mode):
    e1, e3, e_conv = OLD['c222']
    mp = run(c222, max_cycle=1, diis=False, kpt_symmetry=mode)
    assert mp._sym.nibz == (3 if mode else 8)
    assert abs(mp.e_tot - e1) < 1e-8
    assert abs(run(c222, max_cycle=3, diis=False, kpt_symmetry=mode).e_tot - e3) < 1e-8
    mp = run(c222, kpt_symmetry=mode)
    assert mp.converged and abs(mp.e_tot - e_conv) < 1e-8
    assert len(mp.mo_energy) == 8 and numpy.asarray(mp.get_fock()).shape == (8, 8, 8)


@pytest.mark.slow
def test_ipea_222(c222):
    '''The EA of the 0.2 code (its IP misses part of the exchange terms), and
    the same quasiparticle gap with and without k-point symmetry.'''
    a = run(c222, conv_tol_grad=1e-7, kpt_symmetry=True)
    b = run(c222, conv_tol_grad=1e-7, kpt_symmetry=False)
    ga, gb = a.ipea(), b.ipea()
    assert abs(a.ea[0][0] - (-0.973201715743539)) < 1e-6
    assert abs(ga - gb) < 1e-7
    for x, y in zip(a.ip, b.ip):
        assert abs(numpy.sort(x)[-1] - numpy.sort(y)[-1]) < 1e-7


@pytest.mark.slow
def test_symmetry_modes():
    '''Mesh with k-points that are not time-reversal invariant, low-symmetry
    cell: the three symmetry modes give the same energy and bands.'''
    mf = krhf(diamond(dx=0.15), [3, 3, 1])
    ref = run(mf, kpt_symmetry=False)
    assert ref._sym.nibz == 9
    for mode in ('time_reversal', True):
        mp = run(mf, kpt_symmetry=mode)
        assert mp._sym.nibz < 9
        assert abs(mp.e_tot - ref.e_tot) < 1e-8
        for a, b in zip(mp.mo_energy, ref.mo_energy):
            assert abs(numpy.sort(a) - numpy.sort(b)).max() < 1e-6
    a = run(mf, frozen=1, kpt_symmetry=False)
    b = run(mf, frozen=1, kpt_symmetry=True)
    assert abs(a.e_tot - b.e_tot) < 1e-8
    assert a.e_tot > ref.e_tot


@pytest.mark.slow
def test_occupations_depend_on_kpoint():
    '''Hydrogen lattice with one electron per cell: half of the k-points have
    no occupied orbital.'''
    cell = gto.M(atom='H 0 0 0', a=numpy.eye(3) * 2.2 + 0.25 * (1 - numpy.eye(3)),
                 basis='gth-dzv', pseudo='gth-pade', verbose=0)
    mf = krhf(cell, [2, 2, 2])
    nocc = [int((o > 0).sum()) for o in mf.mo_occ]
    assert min(nocc) == 0 and max(nocc) == 1
    pt = pmp.KMP2(mf)
    pt.kernel()
    for mode in (False, True):
        assert abs(run(mf, max_cycle=1, kpt_symmetry=mode).e_tot - mf.e_tot - pt.e_corr) < 1e-8
    assert abs(run(mf, kpt_symmetry=True).e_tot - run(mf, kpt_symmetry=False).e_tot) < 1e-8


@pytest.mark.slow
def test_symmetry_adapted_mean_field():
    cell = boron_nitride()
    full = run(krhf(cell, [2, 2, 2]))
    cell = cell.copy()
    cell.space_group_symmetry = True
    cell.symmorphic = True
    cell.build()
    kpts = cell.make_kpts([2, 2, 2], space_group_symmetry=True, time_reversal_symmetry=True)
    mf = scf.KRHF(cell, kpts).density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    assert abs(run(mf).e_tot - full.e_tot) < 1e-8


@pytest.mark.slow
def test_fft_mean_field():
    '''Mean field with FFTDF: the correlation part builds its own GDF.'''
    from pyscf.pbc import df
    cell = diamond()
    cell.mesh = [13, 13, 13]
    cell.build()
    mf = scf.KRHF(cell, cell.make_kpts([1, 1, 2]))
    mf.conv_tol = 1e-10
    mf.kernel()
    mp = run(mf)
    assert isinstance(mp.with_df, df.GDF) and mp.converged
    # the previous implementation, with FFTDF integrals throughout: -10.640778
    assert abs(mp.e_tot - (-10.6400)) < 2e-3


_MPI_SCRIPT = '''
import sys, warnings
warnings.filterwarnings('ignore')
from pyscf.pbc import gto, scf
from pycmf.core import mpi
from pycmf.pbc import KOBMP2
a0 = 3.5668
cell = gto.M(atom='C 0 0 0; C %f %f %f' % (a0 / 4, a0 / 4, a0 / 4),
             a=[[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]],
             basis='gth-szv', pseudo='gth-pade', verbose=0)
mf = scf.KRHF(cell, cell.make_kpts([2, 2, 2])).density_fit()
mf.conv_tol = 1e-11
mf.kernel()
for mode in (False, True):
    mp = KOBMP2(mf, kpt_symmetry=mode)
    mp.conv_tol = 1e-10
    mp.kernel()
    if mpi.rank == 0:
        print('RESULT %d %.12f' % (mpi.size, mp.e_tot))
'''


@pytest.mark.slow
def test_mpi(tmp_path):
    pytest.importorskip('mpi4py')
    mpiexec = shutil.which('mpiexec') or shutil.which('mpirun')
    if mpiexec is None:
        pytest.skip('no MPI launcher')
    script = tmp_path / 'run.py'
    script.write_text(_MPI_SCRIPT)
    src = os.path.dirname(os.path.dirname(os.path.abspath(pycmf.__file__)))
    env = dict(os.environ, OMP_NUM_THREADS='1',
               PYTHONPATH=os.pathsep.join([src, os.environ.get('PYTHONPATH', '')]))
    out = subprocess.run([mpiexec, '-n', '2', sys.executable, str(script)], env=env,
                         capture_output=True, text=True, timeout=1200)
    assert out.returncode == 0, out.stderr[-2000:]
    res = [line.split() for line in out.stdout.splitlines() if line.startswith('RESULT')]
    assert len(res) == 2 and all(r[1] == '2' for r in res)
    for r in res:
        assert abs(float(r[2]) - OLD['c222'][2]) < 1e-8
