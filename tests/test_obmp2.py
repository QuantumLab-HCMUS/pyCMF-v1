import numpy
import pytest
from pyscf import mp

import pycmf

# total energies from the original implementation (see conftest.py)
REF_U = {  # molecule: (BCH2, BCH1, level_shift=0.2, DF-HF + DF)
    'h2o': (-76.11339945984483, -76.10883825268968, -76.11251428526013, -76.11337179815428),
    'oh': (-75.45274339709474, -75.45016659667473, -75.45221420388623, -75.45272186657368),
    'ch2': (-38.96531705279565, -38.964106577292874, -38.9649045567139, -38.9652893043716),
}
REF_R = {'h2o': (-76.1133994598005, -76.10883825262202)}
TOL = 5e-9


@pytest.mark.parametrize('name', ['h2o', 'oh', 'ch2'])
def test_first_cycle_is_mp2(mf_cache, name):
    mf = mf_cache(name)
    m = pycmf.OBMP2(mf, max_cycle=1)
    m.verbose = 0
    assert abs(m.kernel() - mp.UMP2(mf).run().e_tot) < 1e-9
    assert not m.converged


def test_first_cycle_is_mp2_restricted(mf_cache):
    mf = mf_cache('h2o', 'rhf')
    m = pycmf.OBMP2(mf, max_cycle=1)
    m.verbose = 0
    assert abs(m.kernel() - mp.MP2(mf).run().e_tot) < 1e-9


def test_first_cycle_is_dfmp2(mf_cache):
    mf = mf_cache('h2o', 'rhf', df=True)
    m = pycmf.OBMP2(mf, max_cycle=1)
    m.verbose = 0
    assert abs(m.kernel() - mp.MP2(mf).run().e_tot) < 1e-9


@pytest.mark.parametrize('name', ['h2o', 'oh', 'ch2'])
def test_regression_unrestricted(mf_cache, name):
    mf = mf_cache(name)
    e2, e1, eshift, _ = REF_U[name]
    m = pycmf.OBMP2(mf, conv_tol=1e-10).run()
    assert m.converged
    assert isinstance(m, pycmf.UOBMP2)
    assert abs(m.e_tot - e2) < TOL
    assert abs(m.e_corr - (e2 - mf.e_tot)) < TOL
    assert abs(pycmf.OBMP2(mf, bch_order=1, conv_tol=1e-10).kernel() - e1) < TOL
    assert abs(pycmf.OBMP2(mf, level_shift=0.2, conv_tol=1e-10).kernel() - eshift) < TOL


@pytest.mark.parametrize('name', ['h2o', 'oh', 'ch2'])
def test_regression_density_fitting(mf_cache, name):
    mf = mf_cache(name, df=True)
    m = pycmf.OBMP2(mf, conv_tol=1e-10).run()
    assert m.with_df is mf.with_df
    assert abs(m.e_tot - REF_U[name][3]) < TOL


def test_regression_restricted(mf_cache):
    mf = mf_cache('h2o', 'rhf')
    m = pycmf.OBMP2(mf, conv_tol=1e-10).run()
    assert isinstance(m, pycmf.ROBMP2)
    assert abs(m.e_tot - REF_R['h2o'][0]) < TOL
    assert abs(pycmf.OBMP2(mf, bch_order=1, conv_tol=1e-10).kernel() - REF_R['h2o'][1]) < TOL


def test_restricted_equals_unrestricted(mf_cache):
    er = pycmf.OBMP2(mf_cache('h2o', 'rhf'), conv_tol=1e-10).kernel()
    eu = pycmf.OBMP2(mf_cache('h2o', 'uhf'), conv_tol=1e-10).kernel()
    assert abs(er - eu) < 1e-8


def test_ri_auxiliary_basis(mf_cache):
    '''density_fit() on an exact-integral reference uses an RI-C basis.'''
    mf = mf_cache('h2o', 'rhf')
    e_exact = pycmf.OBMP2(mf).kernel()
    m = pycmf.OBMP2(mf).density_fit()
    assert m.with_df is not None
    assert abs(m.kernel() - e_exact) < 5e-4


def test_brillouin_condition(mf_cache):
    m = pycmf.OBMP2(mf_cache('oh'), conv_tol=1e-10, conv_tol_grad=1e-7).run()
    for f, occ in zip(m.get_fock(), m.mo_occ):
        n = int(occ.sum())
        assert numpy.linalg.norm(f[:n, n:]) < 1e-7


def test_orbital_energies_are_fock_eigenvalues(mf_cache):
    m = pycmf.OBMP2(mf_cache('h2o', 'rhf'), conv_tol=1e-11, conv_tol_grad=1e-8).run()
    f = m.get_fock()
    assert abs(numpy.diag(f) - m.mo_energy).max() < 1e-6
    assert abs(f - numpy.diag(numpy.diag(f))).max() < 1e-6


def test_reference_is_not_modified(mf_cache):
    mf = mf_cache('oh')
    c, e, o = mf.mo_coeff.copy(), mf.mo_energy.copy(), mf.mo_occ.copy()
    pycmf.OBMP2(mf).run()
    assert numpy.array_equal(c, mf.mo_coeff)
    assert numpy.array_equal(e, mf.mo_energy)
    assert numpy.array_equal(o, mf.mo_occ)


def test_rohf_reference_is_converted(mf_cache):
    m = pycmf.OBMP2(mf_cache('oh', 'rohf'), conv_tol=1e-10).run()
    assert isinstance(m, pycmf.UOBMP2)
    assert abs(m.e_tot - REF_U['oh'][0]) < 1e-7


def test_no_diis_gives_same_energy(mf_cache):
    mf = mf_cache('h2o', 'rhf')
    e = pycmf.OBMP2(mf, diis=False, conv_tol=1e-10, max_cycle=200).kernel()
    assert abs(e - REF_R['h2o'][0]) < 1e-8


def test_input_checks(mf_cache):
    mf = mf_cache('h2o', 'rhf')
    with pytest.raises(NotImplementedError):
        pycmf.OBMP2(mf, frozen=1)
    with pytest.raises(ValueError):
        pycmf.OBMP2(mf, bch_order=3).kernel()
    with pytest.raises(ValueError):
        pycmf.OBMP2(mf, occupation='lowest').kernel()
    with pytest.raises(TypeError):
        pycmf.ROBMP2(mf_cache('oh'))


def test_one_electron_system():
    '''No beta electron: the empty spin channel must not break the DF path.'''
    from pyscf import gto, scf
    mol = gto.M(atom='H 0 0 0; H 0 0 1.06', charge=1, spin=1, basis='def2-svp', verbose=0)
    for df in (False, True):
        mf = scf.UHF(mol)
        mf = (mf.density_fit() if df else mf).run(conv_tol=1e-11)
        assert abs(pycmf.OBMP2(mf, conv_tol=1e-10).kernel() - mf.e_tot) < 1e-10
