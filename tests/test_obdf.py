import pytest
from pyscf import fci, scf
from pyscf.tools import fcidump

import pycmf
from conftest import make_mol

# (molecule, basis, 0-based active orbitals, active electrons, E_OBMP2, E_OBDF)
# from the original implementation
CASES = [
    ('H 0 0 0; H 0 0 0.74', 'cc-pvdz', [0, 1, 4, 5], (1, 1), -1.1551137261293887, -1.1581323281284668),
    ('Li 0 0 0; H 0 0 1.6', '6-31g', [1, 2, 3, 4, 5], (1, 1), -7.99224469218855, -7.992696864855882),
    ('Li 0 0 0; H 0 0 1.6', '6-31g', [0, 1, 2, 3, 4, 5], (2, 2), -7.9922446921885575, -7.992692804121868),
    ('O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692', '6-31g', [2, 3, 4, 5, 6, 7], (3, 3),
     -76.11339945980053, -76.11578285112633),
    ('O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692', '6-31g', [2, 4, 5, 6, 8], (2, 2),
     -76.11339945980046, -76.11658747872274),
]


@pytest.mark.parametrize('case', CASES, ids=lambda c: '%s-%s' % (c[0][:2].strip(), len(c[2])))
@pytest.mark.parametrize('kind', ['rhf', 'uhf'])
def test_regression(case, kind):
    from pyscf import gto
    atom, basis, cas, nelec, e_ob, e_df = case
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = (scf.RHF if kind == 'rhf' else scf.UHF)(mol).run(conv_tol=1e-12)
    mp = pycmf.OBMP2(mf, conv_tol=1e-10).run()
    assert abs(mp.e_tot - e_ob) < 5e-9
    ham = pycmf.OBDF(mp, len(cas), nelec, cas_list=cas)
    assert abs(ham.kernel() - e_df) < 5e-8


@pytest.fixture(scope='module')
def lih():
    mf = scf.RHF(make_mol('lih')).run(conv_tol=1e-12)
    return mf, pycmf.OBMP2(mf, conv_tol=1e-10).run()


def test_full_space_is_fci(lih):
    mf, mp = lih
    mol = mf.mol
    e = pycmf.OBDF(mp, mol.nao, mol.nelectron).kernel()
    assert abs(e - fci.FCI(mf).kernel()[0]) < 1e-9


def test_empty_space_is_obmp2(lih):
    _, mp = lih
    ham = pycmf.OBDF(mp, 0, 0).build()
    assert abs(ham.e_core - mp.e_tot) < 1e-10


def test_default_active_space(lih):
    _, mp = lih
    a = pycmf.OBDF(mp, 5, 2)
    b = pycmf.OBDF(mp, 5, (1, 1), cas_list=[1, 2, 3, 4, 5])
    assert abs(a.kernel() - b.kernel()) < 1e-10
    h1, e_core = a.get_h1eff()
    assert h1.shape == (5, 5) and a.get_h2eff().shape == (5,) * 4
    assert abs(h1 - h1.T).max() < 1e-12


def test_fcidump_roundtrip(lih, tmp_path):
    _, mp = lih
    ham = pycmf.OBDF(mp, 5, 2)
    e = ham.kernel()
    fname = str(tmp_path / 'lih.fcidump')
    ham.to_fcidump(fname)
    data = fcidump.read(fname, verbose=0)
    e2 = fci.direct_spin1.kernel(data['H1'], data['H2'], data['NORB'], data['NELEC'])[0]
    assert abs(e2 + data['ECORE'] - e) < 1e-10


def test_excited_roots(lih):
    _, mp = lih
    ham = pycmf.OBDF(mp, 5, 2)
    ham.nroots = 3
    e = ham.kernel()
    assert len(e) == 3 and e[0] < e[1] <= e[2]


def test_density_fitting(mf_cache):
    mp = pycmf.OBMP2(mf_cache('h2o', 'rhf', df=True), conv_tol=1e-10).run()
    e_df = pycmf.OBDF(mp, 6, 6).kernel()
    assert abs(e_df - CASES[3][5]) < 2e-4


def test_unrestricted_open_shell(mf_cache):
    mp = pycmf.OBMP2(mf_cache('oh'), conv_tol=1e-10).run()
    ham = pycmf.OBDF(mp, 5, (3, 2))
    e = ham.kernel()
    assert e < mp.e_tot + 1e-3
    assert len(ham.get_h2eff()) == 3
    with pytest.raises(NotImplementedError):
        ham.to_fcidump('unused')


def test_input_checks(lih, mf_cache):
    _, mp = lih
    with pytest.raises(ValueError):
        pycmf.OBDF(mp, 3, 2, cas_list=[1, 2]).build()
    with pytest.raises(RuntimeError):
        pycmf.OBDF(pycmf.OBMP2(mf_cache('h2o', 'rhf')), 2, 2)
    with pytest.raises(NotImplementedError):
        pycmf.OBDF(pycmf.OBDH(mf_cache('h2o', 'rhf', df=True)).run(), 2, 2)
