'''k-point one-body downfolding (experimental).

References:
  * the four-index implementation of the group (KROBDF, commit 67a4494 of the
    branch ``thai``) with ``exxdiv=None``;
  * the molecular OBDF, fed with the integrals of a Gamma-point cell;
  * exact limits: complete and empty active space, the sum rule for the
    reference determinant, and conventional second-order quasiparticle
    energies for a complete active space.
'''

import numpy
import pytest
from pyscf import ao2mo
from pyscf import gto as mgto
from pyscf import scf as mscf
from pyscf.pbc import df, gto, scf

import pycmf
from pycmf.core.status import ExperimentalWarning
from pycmf.pbc import KOBMP2, KOBDF, kqp

pytestmark = pytest.mark.filterwarnings('ignore::pycmf.core.status.ExperimentalWarning')

A0 = 3.5668

# E(KOBMP2), E_core, trace of h1eff per k-point and the mean-field bands of the
# active-space Hamiltonian at the first two k-points (KROBDF, exxdiv=None)
OLD = {
    'h4': (-1.8577467828991936, -1.2282905306754217,
           [0.6860427167, 0.7753662591, 0.6760222221, 0.8111711979],
           [[-0.1856524062, 0.1253401942, 0.5274317563, 0.7470487765],
            [-0.3310565192, 0.3651235258, 0.5047570954, 0.8092306389]]),
    'c': (-9.413693936297635, -12.347625178497342,
          [5.1571168128, 5.0675247533, 5.0675247533, 4.9254388513],
          [[0.6582601979, 0.7027490884, 0.7168707305, 1.0388354182, 1.0439138505, 1.0721817317],
           [0.1280817694, 0.5343777134, 0.5753521877, 1.2241473399, 1.2312885488, 1.2758880653]]),
}
SYSTEMS = {'h4': ([2, 1, 2], 4, 2), 'c': ([2, 2, 1], 6, 6)}


def make_cell(name):
    if name == 'h4':
        return gto.M(atom='H 0 0 0; H 0 0 0.8; H 0.3 0.2 1.9; H 0.1 0.4 3.0',
                     a=[[4.5, 0, 0], [0, 4.6, 0], [0.3, 0.2, 4.3]],
                     basis='gth-dzv', pseudo='gth-pade', verbose=0)
    return gto.M(atom='C 0 0 0; C %f %f %f' % (A0 / 4, A0 / 4, A0 / 4),
                 a=[[0, A0 / 2, A0 / 2], [A0 / 2, 0, A0 / 2], [A0 / 2, A0 / 2, 0]],
                 basis='gth-szv', pseudo='gth-pade', verbose=0)


@pytest.fixture(scope='module')
def kmf_cache():
    cache = {}

    def get(name, exxdiv=None):
        if (name, exxdiv) not in cache:
            cell = make_cell(name)
            kpts = cell.make_kpts(SYSTEMS[name][0])
            mf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
            mf.with_df = df.GDF(cell, kpts)
            mf.conv_tol = 1e-11
            mf.kernel()
            cache[name, exxdiv] = mf
        return cache[name, exxdiv]
    return get


@pytest.fixture(scope='module')
def mp_cache(kmf_cache):
    cache = {}

    def get(name, exxdiv=None, **kwargs):
        key = (name, exxdiv, tuple(sorted(kwargs.items())))
        if key not in cache:
            mp = KOBMP2(kmf_cache(name, exxdiv))
            mp.conv_tol = 1e-10
            mp.__dict__.update(kwargs)
            mp.kernel()
            cache[key] = mp
        return cache[key]
    return get


@pytest.mark.parametrize('name', ['h4', 'c'])
def test_previous_implementation(mp_cache, name):
    e_mp, e_core, trace, bands = OLD[name]
    _, ncas, nelecas = SYSTEMS[name]
    mp = mp_cache(name)
    assert abs(mp.e_tot - e_mp) < 1e-8
    with pytest.warns(ExperimentalWarning):
        ham = KOBDF(mp, ncas, nelecas)
    h1, ec = ham.get_h1eff()
    assert h1.shape == (4, ncas, ncas) and abs(h1 - h1.conj().transpose(0, 2, 1)).max() < 1e-12
    assert abs(ec - e_core) < 1e-7
    assert abs(numpy.einsum('kii->k', h1).real - trace).max() < 1e-7
    ham.scf()
    assert ham.converged
    assert abs(ham.mo_energy[:2] - numpy.array(bands)).max() < 1e-7
    # four-index integrals from the three-index ones
    kconserv = mp._kconserv()
    c = ham.mo_coeff
    kpts = mp.kpts
    k4 = kconserv[1, 2, 3]
    eri = mp._scf.with_df.ao2mo((c[1], c[2], c[3], c[k4]), (kpts[1], kpts[2], kpts[3], kpts[k4]),
                                compact=False).reshape((ncas,) * 4) / 4
    assert abs(ham.get_h2eff(1, 2, 3) - eri).max() < 1e-9
    assert abs(ham.get_h2eff()[1, 2, 3] - eri).max() < 1e-9


@pytest.mark.parametrize('exxdiv', [None, 'ewald'])
def test_limits_and_sum_rule(mp_cache, exxdiv):
    mp = mp_cache('h4', exxdiv)
    cell = mp.cell
    nmo = mp.mo_coeff[0].shape[1]
    # complete active space: the Hamiltonian itself
    full = mp.downfold(nmo, cell.nelectron)
    assert abs(full.energy_tot() - mp.e_ref) < 1e-10
    assert abs(full.e_corr_act - (mp.e_tot - mp.e_ref)) < 1e-10
    assert max(abs(a - b).max() for a, b in zip(full.get_fock(), mp.fock_hf)) < 1e-9
    # empty active space: the KOBMP2 energy
    assert abs(KOBDF(mp, 0, 0).get_h1eff()[1] - mp.e_tot) < 1e-10
    # reference determinant of any active space
    for ncas, nelecas in ((4, 2), (3, 4), (2, 2)):
        ham = KOBDF(mp, ncas, nelecas)
        assert abs(ham.energy_tot() - (mp.e_tot - ham.e_corr_act)) < 1e-10
        e_hf = ham.scf()
        assert ham.converged and e_hf < ham.energy_tot(ham.reference_rdm1()) + 1e-12
    with pytest.raises(ValueError):
        KOBDF(mp, 4, 2, cas_list=[0, 1, 2, 3]).build()
    with pytest.raises(ValueError):
        KOBDF(mp, 4, 3)


def test_kpoint_symmetry_and_spin_scaling(kmf_cache):
    out = []
    for mode in (True, False):
        mp = KOBMP2(kmf_cache('c'), kpt_symmetry=mode)
        mp.conv_tol = 1e-10
        mp.c_os, mp.c_ss = 1.2, 1. / 3
        mp.kernel()
        ham = KOBDF(mp, 6, 6)
        h1, ec = ham.get_h1eff()
        out.append((mp._sym.nibz, ec, numpy.array([numpy.linalg.eigvalsh(x) for x in h1]), ham.scf(),
                    ham.energy_tot(ham.reference_rdm1()) - (mp.e_tot - ham.e_corr_act)))
    assert out[0][0] == 3 and out[1][0] == 4
    assert abs(out[0][1] - out[1][1]) < 1e-8
    assert abs(out[0][2] - out[1][2]).max() < 1e-8
    assert abs(out[0][3] - out[1][3]) < 1e-8
    assert abs(out[0][4]) < 1e-9 and abs(out[1][4]) < 1e-9


@pytest.mark.parametrize('exxdiv', [None, 'ewald'])
def test_supercell_solvers(mp_cache, exxdiv):
    mp = mp_cache('h4', exxdiv)
    ham = KOBDF(mp, 3, 2)
    h1, h2, e0, nelec = ham.supercell()
    assert h1.shape == (12, 12) and nelec == 8 and abs(e0 - 4 * ham.e_core) < 1e-12
    assert abs(h1 - h1.T).max() < 1e-9
    assert abs(h2 - h2.transpose(1, 0, 3, 2)).max() < 1e-9 and abs(h2 - h2.transpose(2, 3, 0, 1)).max() < 1e-9
    e_hf = ham.scf()
    mf = ham.as_scf().run(conv_tol=1e-12)
    assert abs(mf.e_tot / 4 - e_hf) < 1e-8
    e_fci = ham.kernel()
    assert e_fci < e_hf and abs(ham.e_cas + ham.e_core - e_fci) < 1e-12
    # the supercell Hamiltonian is that of the k-space integrals
    from pyscf import cc
    e_cc = cc.CCSD(mf).run(conv_tol=1e-10).e_tot / 4
    assert e_fci - 1e-8 < e_cc < e_hf


def test_quasiparticle_complete_space_is_second_order_of_hf(kmf_cache):
    '''HF orbitals and a complete active space: H_act is the Hamiltonian, its
    mean field is KRHF and its self-energy the conventional second-order one.'''
    mp = KOBMP2(kmf_cache('h4')).init_from_scf()
    nmo, nocc = mp.mo_coeff[0].shape[1], mp.cell.nelectron // 2
    ham = KOBDF(mp, nmo, 2 * nocc)
    e_hf = ham.scf()
    assert abs(e_hf - mp._scf.e_tot) < 1e-9
    assert abs(ham.mo_energy - numpy.asarray(mp._scf.mo_energy)).max() < 1e-7
    se = mp.self_energy(nvir=2)
    ham.self_energy(nvir=2)
    for scheme in ('static', 'linear', 'dynamic'):
        ref = kqp.gaps(se, kqp.quasiparticle(se, scheme, 'hf')[0])[:2]
        assert abs(numpy.array(ham.quasiparticle_gap(scheme)) - ref).max() < 1e-7
    # spin-component scaling of the self-energy
    mp.c_os, mp.c_ss = 1.3, 0.
    se = mp.self_energy(nvir=1, nocc=1)
    ham.self_energy(nocc=1, nvir=1, c_os=1.3, c_ss=0.)
    ref = kqp.gaps(se, kqp.quasiparticle(se, 'static', 'hf')[0])[:2]
    assert abs(numpy.array(ham.quasiparticle_gap('static')) - ref).max() < 1e-7


def test_quasiparticle_of_the_downfolded_hamiltonian(mp_cache):
    '''Regression of the present implementation: mean-field gap of the model
    and its second-order corrections (small for this weakly correlated cell).'''
    mp = mp_cache('h4')
    ham = mp.downfold(4, 2)
    ham.scf()
    ham.self_energy()
    gaps = {s: ham.quasiparticle_gap(s)[1] for s in ('static', 'linear', 'dynamic')}
    assert abs(ham.band_gap() - 0.3097391195) < 1e-7
    assert abs(gaps['linear'] - gaps['dynamic']) < 1e-4
    assert all(abs(g - ham.band_gap()) < 5e-3 for g in gaps.values())


@pytest.mark.parametrize('opts', [{}, {'c_os': 1.2, 'c_ss': 0.}, {'bch_order': 1}])
def test_gamma_point_equals_molecule(opts, gamma_pair):
    kmf, mf = gamma_pair
    a = pycmf.OBMP2(mf)
    a.__dict__.update(opts)
    a.conv_tol = 1e-11
    a.kernel()
    b = KOBMP2(kmf)
    b.__dict__.update(opts)
    b.conv_tol = 1e-11
    b.kernel()
    for ncas, nelecas in ((4, 4), (5, 6)):
        m = pycmf.OBDF(a, ncas, nelecas)
        k = KOBDF(b, ncas, nelecas)
        hm, em = m.get_h1eff()
        hk, ek = k.get_h1eff()
        assert abs(em - ek) < 2e-8
        assert abs(numpy.linalg.eigvalsh(hm) - numpy.linalg.eigvalsh(hk[0])).max() < 2e-8
        assert abs(m.kernel() - k.kernel()) < 2e-8


@pytest.fixture(scope='module')
def gamma_pair():
    a = 3.615
    cell = gto.M(atom='B 0 0 0; N %.10f %.10f %.10f' % (a / 4 + .05, a / 4, a / 4 - .03),
                 a=[[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]],
                 basis='gth-szv', pseudo='gth-pade', verbose=0)
    kmf = scf.KRHF(cell, cell.make_kpts([1, 1, 1]), exxdiv=None).density_fit()
    kmf.conv_tol = 1e-11
    kmf.kernel()
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
