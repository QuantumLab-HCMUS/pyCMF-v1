'''State-specific calculations started from a non-Aufbau determinant.'''

import pytest
from pyscf import scf

import pycmf
from pycmf.core import occupation
from conftest import make_mol

# original implementation, alpha HOMO -> LUMO excited determinant of water
E_MOM_UHF = -75.71282685909686
E_OBMP2 = -75.8094975850136
E_O2BMP2 = -75.78376321621603


@pytest.fixture(scope='module')
def mf_excited():
    mol = make_mol('h2o')
    mf0 = scf.UHF(mol).run(conv_tol=1e-12)
    occ = mf0.mo_occ.copy()
    occ[0][4], occ[0][5] = 0, 1
    mf = scf.addons.mom_occ(scf.UHF(mol), mf0.mo_coeff, occ)
    mf.conv_tol = 1e-12
    mf.kernel(mf.make_rdm1(mf0.mo_coeff, occ))
    assert abs(mf.e_tot - E_MOM_UHF) < 1e-7
    return mf


@pytest.mark.parametrize('scheme', ['imom', 'mom'])
def test_excited_state(mf_excited, scheme):
    m = pycmf.OBMP2(mf_excited, occupation=scheme, conv_tol=1e-10).run()
    assert m.converged
    assert abs(m.e_tot - E_OBMP2) < 1e-7
    assert m.spin_square()[0] > 0.9  # spin-mixed open-shell determinant


def test_excited_state_o2bmp2(mf_excited):
    m = pycmf.O2BMP2(mf_excited, c_os=1.2, occupation='imom', conv_tol=1e-10).run()
    assert abs(m.e_tot - E_O2BMP2) < 1e-7


def test_non_aufbau_start_is_reordered(mf_cache):
    '''A determinant given by mo_occ is used even if it is not Aufbau.'''
    mf = mf_cache('h2o')
    occ = mf.mo_occ.copy()
    occ[0][4], occ[0][5] = 0, 1
    m = pycmf.OBMP2(mf, mo_occ=occ, max_cycle=1, occupation='imom')
    m.verbose = 0
    m.kernel()
    assert m.e_ref > mf.e_tot + 0.1


def test_select():
    import numpy
    e = numpy.arange(4.)
    c = numpy.eye(4)
    ref = c[:, [0, 2]]
    e2, c2 = occupation.select(e, c, 2, 'imom', ref, numpy.eye(4))
    assert list(e2) == [0., 2., 1., 3.]
    assert occupation.lost_overlap(c2, 2, ref, numpy.eye(4)) < 1e-12
