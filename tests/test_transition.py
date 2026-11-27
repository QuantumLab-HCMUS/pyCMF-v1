'''Transition dipole moments between separately optimized determinants.'''

import warnings

import numpy
import pytest
from pyscf import gto, scf

import pycmf
from pycmf.core.status import ExperimentalWarning
from pycmf.mol import excited
from pycmf.properties import transition
from fci_reference import FCISpace, loewdin_pair

pytestmark = pytest.mark.filterwarnings('ignore::pycmf.core.status.ExperimentalWarning')

# H2O of the CC3 benchmark set (xyz/Mol/11-H2O.xyz of the OB-O2B scripts)
H2O_CC3 = '''O 0.0000000000 0.0000000000 -0.0699025580
H 0.0000000000 0.7575324071 0.5184349453
H 0.0000000000 -0.7575324071 0.5184349453'''

# Original OB-O2B-simple scripts (calc_ob_o2b_transition.py) with the 0.2
# library uobmp2_mom_conv: OBMP2/cc-pVDZ, beta 4 -> 5, sqrt(2) singlet factor
E_GS_REF = -76.23133477534225
E_ES_REF = -75.9345521028487
MU_DET_REF = 0.236223433          # |mu_det| (e a0)
MU_REF = 0.33407038               # sqrt(2) |mu_det|
F_REF = 0.02208123


@pytest.fixture(scope='module')
def distorted():
    '''Water without symmetry: the excited determinants overlap the ground state.'''
    mol = gto.M(atom='O 0 0 0.05; H 0.1 0.78 -0.47; H -0.05 -0.74 -0.45',
                basis='sto-3g', verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1e-12)
    return mol, mf


def test_hf_limit_column_swap(distorted):
    '''An unrelaxed single excitation is orthogonal to the HF determinant and
    <0|mu|i->a> = -<i|r|a>.'''
    mol, mf = distorted
    i, a = 4, 5
    ca, cb = mf.mo_coeff
    occ_a = ca[:, [0, 1, 2, 3, a]]
    es = (occ_a, cb[:, :5])
    gs = (ca[:, :5], cb[:, :5])
    s1e = mol.intor('int1e_ovlp')
    s, (pa, pb) = transition.transition_density(gs, es, s1e)
    assert abs(s) < 1e-12
    dm = transition.orthogonalize(gs[0] @ gs[0].T + gs[1] @ gs[1].T,
                                  es[0] @ es[0].T + es[1] @ es[1].T, pa + pb, s)
    r = mol.intor('int1e_r', comp=3)
    mu = -numpy.einsum('xpq,qp->x', r, dm)
    ref = -numpy.einsum('p,xpq,q->x', ca[:, i], r, ca[:, a])
    assert abs(abs(mu) - abs(ref)).max() < 1e-10


def test_orthogonalized_dipole_against_fci(distorted):
    '''Loewdin orthogonalization in the full CI space gives the same moment.'''
    mol, mf = distorted
    mf_es = excited.mom_excitation(mf, 3, 6, spin=0, conv_tol=1e-12)
    res = transition.transition_dipole(mf, mf_es)
    assert abs(res.overlap) > 1e-3  # a real test of the orthogonalization
    space = FCISpace(mol, mf.mo_coeff[0], mol.nelec)
    cg = space.vector(transition.occupied_orbitals(mf))
    ce = space.vector(transition.occupied_orbitals(mf_es))
    assert abs(space.overlap(cg, ce) - res.overlap) < 1e-10
    g, e = loewdin_pair(cg, ce)
    assert abs(space.overlap(g, e)) < 1e-12
    assert abs(space.dipole(g, e) - res.dipole_det).max() < 1e-9
    assert abs(res.e_exc - (mf_es.e_tot - mf.e_tot)) < 1e-12


def test_symmetric_in_the_two_states(distorted):
    mol, mf = distorted
    mf_es = excited.mom_excitation(mf, 4, 5, spin=1, conv_tol=1e-12)
    a = transition.transition_dipole(mf, mf_es)
    b = transition.transition_dipole(mf_es, mf)
    assert abs(a.dipole - b.dipole).max() < 1e-10
    assert abs(a.e_exc + b.e_exc) < 1e-12
    assert abs(a.f + b.f) < 1e-12


def test_singlet_factor(distorted):
    mol, mf = distorted
    mf_es = excited.mom_excitation(mf, 4, 5, spin=1, conv_tol=1e-12)
    a = transition.transition_dipole(mf, mf_es)
    b = transition.transition_dipole(mf, mf_es, singlet=True)
    assert abs(b.dipole - numpy.sqrt(2) * a.dipole).max() < 1e-14
    assert abs(b.f - 2 * a.f) < 1e-14


def test_restricted_ground_state():
    '''A restricted and an unrestricted ground state give the same result.'''
    mol = gto.M(atom=H2O_CC3, basis='6-31g', verbose=0)
    rhf = scf.RHF(mol).run(conv_tol=1e-12)
    uhf = scf.UHF(mol).run(conv_tol=1e-12)
    mf_es = excited.mom_excitation(rhf, 4, 5, spin=0, conv_tol=1e-12)
    a = transition.transition_dipole(pycmf.OBMP2(rhf, conv_tol=1e-10).run(),
                                     pycmf.OBMP2(mf_es, occupation='imom', conv_tol=1e-10).run())
    b = transition.transition_dipole(pycmf.OBMP2(uhf, conv_tol=1e-10).run(),
                                     pycmf.OBMP2(mf_es, occupation='imom', conv_tol=1e-10).run())
    assert abs(a.e_exc - b.e_exc) < 1e-8
    assert abs(abs(a.dipole) - abs(b.dipole)).max() < 1e-6


def test_regression_obmp2_h2o():
    '''Reproduces calc_ob_o2b_transition.py with the 0.2 library.'''
    mol = gto.M(atom=H2O_CC3, basis='cc-pvdz', verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    gs = pycmf.OBMP2(mf, conv_tol=1e-10).run()
    es = pycmf.OBMP2(excited.mom_excitation(mf, 4, 5, spin=1), occupation='imom',
                     conv_tol=1e-10).run()
    assert gs.converged and es.converged
    assert abs(gs.e_tot - E_GS_REF) < 1e-8
    assert abs(es.e_tot - E_ES_REF) < 1e-8
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        res = transition.transition_dipole(gs, es, singlet=True)
    assert any(issubclass(x.category, ExperimentalWarning) for x in w)
    assert abs(numpy.linalg.norm(res.dipole_det) - MU_DET_REF) < 1e-7
    assert abs(numpy.linalg.norm(res.dipole) - MU_REF) < 1e-7
    assert abs(res.f - F_REF) < 1e-7
    assert abs(res.spin_square[1] - 1.006951) < 1e-5


def test_identical_states_raise(distorted):
    mol, mf = distorted
    with pytest.raises(ValueError):
        transition.transition_dipole(mf, mf)


def test_mom_excitation_checks(distorted):
    mol, mf = distorted
    with pytest.raises(ValueError):
        excited.mom_excitation(mf, 5, 6)      # hole is empty
    with pytest.raises(ValueError):
        excited.mom_excitation(mf, 3, 4)      # particle is occupied
    with pytest.raises(ValueError):
        excited.mom_excitation(mf, 3, 5, spin=2)
    occ_before = mf.mo_occ.copy()
    excited.mom_excitation(mf, 4, 5)
    assert numpy.array_equal(mf.mo_occ, occ_before)


def test_select_channels(distorted):
    mol, mf = distorted
    ch = excited.select_channels(mf, 3)
    assert ch[0][:2] == (4, 5)
    gaps = [c[2] for c in excited.select_channels(mf)]
    assert gaps == sorted(gaps) and len(gaps) == 5 * 2
