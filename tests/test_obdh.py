import pytest
from pyscf import dft

import pycmf
from conftest import make_mol

# original implementation (unrestricted, density fitting, 6-31G):
# (a_x, a_c) = (0.53, 0.27); same with c_ss = 0, c_os = 1.2; (0.5, 0.5)
REF = {
    'h2o': (-76.28399611310215, -76.27182960981932, -76.23591478980386),
    'ch2': (-39.07642841907034, -39.071724623308086, -39.03894810170471),
    'nh2': (-55.76508010088324, -55.757516045591174, -55.72030774102084),
}
TOL = 2e-8


@pytest.mark.parametrize('name', ['h2o', 'ch2', 'nh2'])
def test_regression(mf_cache, name):
    mf = mf_cache(name, df=True)
    m = pycmf.OBDH(mf, conv_tol=1e-10).run()
    assert m.converged and (m.ax, m.ac) == (0.53, 0.27)
    assert abs(m.e_tot - REF[name][0]) < TOL
    assert abs(pycmf.O2BDH(mf, c_os=1.2, conv_tol=1e-10).kernel() - REF[name][1]) < TOL
    assert abs(pycmf.OBDH(mf, ax=0.5, ac=0.5, conv_tol=1e-10).kernel() - REF[name][2]) < TOL


def test_restricted_equals_unrestricted(mf_cache):
    m = pycmf.OBDH(mf_cache('h2o', 'rhf', df=True), conv_tol=1e-10).run()
    assert isinstance(m, pycmf.ROBDH)
    assert abs(m.e_tot - REF['h2o'][0]) < TOL


def test_pure_exchange_and_correlation_is_obmp2(mf_cache):
    mf = mf_cache('h2o', 'rhf', df=True)
    e = pycmf.OBDH(mf, ax=1., ac=1., conv_tol=1e-10).kernel()
    assert abs(e - pycmf.OBMP2(mf, conv_tol=1e-10).kernel()) < 1e-9


def test_no_correlation_weight_is_hybrid_dft(mf_cache):
    mf = mf_cache('h2o', 'rhf', df=True)
    e = pycmf.OBDH(mf, ax=0.2, ac=0., xc_c='LYP', conv_tol=1e-10).kernel()
    ks = dft.RKS(mf.mol, xc='0.2*HF + 0.8*B88, LYP').density_fit()
    ks.with_df = mf.with_df
    assert abs(e - ks.run(conv_tol=1e-11).e_tot) < 1e-7


def test_kohn_sham_starting_orbitals(mf_cache):
    ks = dft.RKS(make_mol('h2o'), xc='0.53*HF + 0.47*B88, 0.73*LYP').density_fit().run()
    assert abs(pycmf.OBDH(ks, conv_tol=1e-10).kernel() - REF['h2o'][0]) < 1e-6


def test_exact_integrals(mf_cache):
    e = pycmf.OBDH(mf_cache('h2o', 'rhf'), conv_tol=1e-9).kernel()
    assert abs(e - REF['h2o'][0]) < 1e-3


def test_functional_string():
    m = pycmf.OBDH(make_mol('h2o').RHF(), ax=0.5, ac=0.25)
    assert m.get_xc() == '0.5*HF + 0.5*B88, 0.75*LYP'
    with pytest.raises(ValueError):
        pycmf.O2BDH(make_mol('h2o').RHF())


def test_named_functionals(mf_cache):
    from pycmf.mol.obdh import FUNCTIONALS, double_hybrid
    assert double_hybrid('B2GP-PLYP') == double_hybrid('b2gpplyp') == (0.65, 0.36, 'B88', 'LYP')
    assert double_hybrid('pbe') is None
    # in every functional of the table the semilocal correlation has the weight 1 - a_c
    mf = mf_cache('h2o', 'rhf', df=True)
    for name, (ax, ac, xc_x, xc_c) in FUNCTIONALS.items():
        m = pycmf.OBDH(mf, xc=name)
        assert (m.ax, m.ac) == (ax, ac) and m.xc is None
        assert m.get_xc() == '%.10g*HF + %.10g*%s, %.10g*%s' % (ax, 1 - ax, xc_x, 1 - ac, xc_c)
    # 'PBE0-2' is the double hybrid, not "PBE0 minus 2" of the libxc parser
    assert pycmf.OBDH(mf, xc='PBE0-2').ac == 0.5
    assert abs(pycmf.OBDH(mf, xc='B2PLYP', conv_tol=1e-10).kernel() - REF['h2o'][0]) < TOL
    e = pycmf.OBDH(mf, xc='PBE0-DH', conv_tol=1e-10).kernel()
    e1 = pycmf.OBDH(mf, ax=0.5, ac=0.125, xc_x='PBE', xc_c='PBE', conv_tol=1e-10).kernel()
    e2 = pycmf.OBDH(mf, xc='0.5*HF + 0.5*PBE, 0.875*PBE', ac=0.125, conv_tol=1e-10).kernel()
    assert abs(e - e1) < 1e-10 and abs(e - e2) < 1e-10
    # explicit parameters override those of the named functional
    assert pycmf.OBDH(mf, xc='B2PLYP', ac=0.3).get_xc() == '0.53*HF + 0.47*B88, 0.7*LYP'
    # the name can also be set as an attribute
    m = pycmf.OBDH(mf)
    m.xc = 'b2gp-plyp'
    assert m._functional() == ('0.65*HF + 0.35*B88, 0.64*LYP', 0.36)
    with pytest.raises(KeyError):
        pycmf.OBDH(mf).set_functional('pbe')
    with pytest.raises(ValueError):
        pycmf.OBDH(mf, xc='pbe0', ax=0.3)
