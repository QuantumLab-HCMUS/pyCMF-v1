'''Orbital-optimized double hybrids (variational in the orbitals).'''

import numpy
import pytest
import scipy.linalg
from pyscf import dft, gto, scf

import pycmf

pytestmark = pytest.mark.filterwarnings('ignore::pycmf.core.status.ExperimentalWarning')

H2O = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
NH2 = 'N 0 0 0.1404; H 0 0.8018 -0.4914; H 0 -0.8018 -0.4914'
DISTORTED = 'O 0 0 0.05; H 0.1 0.78 -0.47; H -0.05 -0.74 -0.45'

# ORCA 4.1.0, cc-pVDZ, cc-pVDZ/C, NoFrozenCore, UKS, VeryTightSCF, Grid7 NoFinalGrid
ORCA = {
    'h2o_ri_mp2': -76.230760436,             # OO-RI-MP2, orbital iteration 0
    'h2o_oo_ri_mp2': -76.231656843479,       # OO-RI-MP2 (gradient tolerance 1e-3)
    'h2o_ri_b2plyp': -76.353769929011,       # RI-B2PLYP
    'h2o_oo_ri_b2plyp': -76.353874683219,    # RI-B2PLYP, %mp2 OrbOpt true
    'nh2_ri_b2plyp': -55.819349497121,
    'nh2_oo_ri_b2plyp': -55.819431774400,
}


def _rotate(c, x, n):
    k = numpy.zeros((c.shape[1],) * 2)
    k[n:, :n] = x
    k[:n, n:] = -x.T
    return c.dot(scipy.linalg.expm(k))


@pytest.mark.parametrize('xc,spin', [('MP2', 0), ('B2PLYP', 0), ('PBE0-DH', 1)])
def test_gradient_finite_difference(xc, spin):
    '''2 fbar_ai is the derivative of the energy, at orbitals away from the minimum.'''
    mol = gto.M(atom=DISTORTED, basis='6-31g', charge=spin, spin=spin, verbose=0)
    mf = dft.UKS(mol, xc='b3lyp').run()
    kw = dict(xc='HF', ac=1.) if xc == 'MP2' else dict(xc=xc)
    m = pycmf.OODH(mf, **kw)
    mo_e, mo_c, nocc = m._init_orbitals()
    rng = numpy.random.default_rng(1)
    mo_c = [_rotate(c, .05 * rng.standard_normal((c.shape[1] - n, n)), n)
            for c, n in zip(mo_c, nocc)]
    im = m.build(mo_e, mo_c, nocc)
    h = 1e-4
    for s, n in enumerate(nocc):
        nvir = mo_c[s].shape[1] - n
        for a, i in [(0, 0), (1, n - 1), (nvir - 1, 1)]:
            e = []
            for d in (h, -h):
                x = numpy.zeros((nvir, n))
                x[a, i] = d
                c = list(mo_c)
                c[s] = _rotate(mo_c[s], x, n)
                e.append(m.build(mo_e, c, nocc).e_tot)
            fd = (e[0] - e[1]) / (2 * h)
            assert abs(2 * im.fock_mo[s][n + a, i] - fd) < 1e-7


def test_oo_mp2_against_orca():
    mol = gto.M(atom=H2O, basis='cc-pvdz', verbose=0)
    hf = scf.UHF(mol).run(conv_tol=1e-12)
    m = pycmf.OODH(hf, xc='HF', ac=1., conv_tol=1e-11, conv_tol_grad=1e-7)
    m.density_fit('cc-pvdz-ri').run()
    assert m.converged
    assert abs(m.e_dh - ORCA['h2o_ri_mp2']) < 1e-8
    assert abs(m.e_tot - ORCA['h2o_oo_ri_mp2']) < 2e-7


@pytest.mark.parametrize('atom,spin,key', [(H2O, 0, 'h2o'), (NH2, 1, 'nh2')])
def test_oo_b2plyp_against_orca(atom, spin, key):
    '''The double hybrid itself differs by a few 1e-7 Eh (functional and grid
    of PySCF and ORCA); the lowering by the orbital optimization agrees to
    1e-8 Eh.'''
    mol = gto.M(atom=atom, basis='cc-pvdz', spin=spin, verbose=0)
    m = pycmf.OODH(mol, xc='B2PLYP', conv_tol=1e-11, conv_tol_grad=1e-7)
    m.density_fit('cc-pvdz-ri').run()
    assert m.converged
    assert abs(m.e_dh - ORCA[key + '_ri_b2plyp']) < 1e-6
    assert abs(m.e_tot - ORCA[key + '_oo_ri_b2plyp']) < 1e-6
    d_orca = ORCA[key + '_oo_ri_b2plyp'] - ORCA[key + '_ri_b2plyp']
    assert abs((m.e_tot - m.e_dh) - d_orca) < 2e-8
    assert m.e_tot < m.e_dh


def test_exact_and_df_integrals():
    mol = gto.M(atom=H2O, basis='6-31g', verbose=0)
    a = pycmf.OODH(mol, conv_tol=1e-10).run()
    b = pycmf.OODH(mol, conv_tol=1e-10).density_fit('weigend').run()
    assert abs(a.e_tot - b.e_tot) < 1e-4
    assert a.e_tot < a.e_dh


def test_hybrid_dft_limit():
    '''ac = 0: the minimum is the Kohn-Sham solution of the hybrid.'''
    mol = gto.M(atom=DISTORTED, basis='6-31g', verbose=0)
    ks = dft.UKS(mol, xc='0.5*HF + 0.5*B88, LYP').run(conv_tol=1e-12)
    m = pycmf.OODH(scf.UHF(mol).run(), ax=.5, ac=0., conv_tol=1e-11, conv_tol_grad=1e-7).run()
    assert abs(m.e_tot - ks.e_tot) < 1e-9
    assert abs(m.e_pt2) > 1e-3     # E_2 is computed but carries no weight


def test_relaxed_density_finite_field():
    '''D + ac gamma is the derivative of the energy (no orbital response).'''
    mol = gto.M(atom=DISTORTED, basis='6-31g', verbose=0)
    m = pycmf.OODH(mol, conv_tol=1e-12, conv_tol_grad=1e-8).run()
    r = mol.intor('int1e_r')
    dm = numpy.asarray(m.make_rdm1(correlated=True)).sum(0)
    mu = -numpy.einsum('pq,qp->', r[2], dm)
    h = 1e-4
    e = []
    for lam in (h, -h):
        mm = pycmf.OODH(m._scf, conv_tol=1e-12, conv_tol_grad=1e-8)
        ks = mm.get_ks()
        h0 = ks.get_hcore()
        ks.get_hcore = lambda *args, h0=h0, lam=lam: h0 + lam * r[2]
        e.append(mm.run().e_tot)
    assert abs(mu + (e[0] - e[1]) / (2 * h)) < 1e-6
    mu_det = -numpy.einsum('pq,qp->', r[2], numpy.asarray(m.make_rdm1()).sum(0))
    assert abs(mu - mu_det) > 1e-3


def test_restricted_start_and_scaling():
    mol = gto.M(atom=H2O, basis='6-31g', verbose=0)
    rks = dft.RKS(mol, xc='b3lyp').run()
    a = pycmf.OODH(rks, conv_tol=1e-10).run()
    b = pycmf.OODH(mol, conv_tol=1e-10).run()
    assert abs(a.e_tot - b.e_tot) < 1e-8
    c = pycmf.OODH(mol, c_os=1.3, c_ss=0., conv_tol=1e-10).run()
    assert c.converged and abs(c.e_tot - b.e_tot) > 1e-4
    with pytest.raises(NotImplementedError):
        pycmf.OODH(mol, level_shift=.1).run()
    with pytest.raises(NotImplementedError):
        a.spin_square(correlated=True)


def test_mom_excited_state():
    '''occupation='imom' keeps a non-Aufbau determinant (state-specific OO-DH).'''
    from pycmf.mol import excited
    mol = gto.M(atom=H2O, basis='6-31g', verbose=0)
    hf = scf.UHF(mol).run()
    es = excited.mom_excitation(hf, 4, 5, spin=1)
    gs = pycmf.OODH(hf, conv_tol=1e-10).run()
    ex = pycmf.OODH(es, occupation='imom', conv_tol=1e-10).run()
    assert ex.converged
    assert 0.2 < ex.e_tot - gs.e_tot < 0.4
