import pytest

import pycmf

# original implementation: (c_ss=0, c_os=1.2), (c_ss=1/3, c_os=1.2)
REF = {
    'h2o': (-76.08030243894648, -76.09696068988555),
    'oh': (-75.42974692384956, -75.44127402182673),
    'ch2': (-38.953179209142235, -38.95912580031201),
}


@pytest.mark.parametrize('name', ['h2o', 'oh', 'ch2'])
def test_regression(mf_cache, name):
    mf = mf_cache(name)
    m = pycmf.O2BMP2(mf, c_os=1.2, conv_tol=1e-10).run()
    assert m.converged and m.c_ss == 0
    assert abs(m.e_tot - REF[name][0]) < 5e-9
    e = pycmf.OBMP2(mf, c_os=1.2, c_ss=1 / 3, conv_tol=1e-10).kernel()
    assert abs(e - REF[name][1]) < 5e-9


def test_restricted_equals_unrestricted(mf_cache):
    er = pycmf.O2BMP2(mf_cache('h2o', 'rhf'), c_os=1.2, conv_tol=1e-10).kernel()
    assert abs(er - REF['h2o'][0]) < 1e-8


def test_unit_scaling_is_obmp2(mf_cache):
    mf = mf_cache('oh')
    e = pycmf.OBMP2(mf, c_os=1., c_ss=1., conv_tol=1e-10).kernel()
    assert abs(e - pycmf.OBMP2(mf, conv_tol=1e-10).kernel()) < 1e-10


def test_no_amplitudes_is_hartree_fock(mf_cache):
    mf = mf_cache('oh')
    m = pycmf.OBMP2(mf, c_os=0., c_ss=0.).run()
    assert abs(m.e_tot - mf.e_tot) < 1e-9


def test_c_os_is_required(mf_cache):
    with pytest.raises(ValueError):
        pycmf.O2BMP2(mf_cache('oh'))
