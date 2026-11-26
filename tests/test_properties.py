import numpy
from pyscf import mp

import pycmf


def test_mp2_like_density_first_cycle(mf_cache):
    '''After one cycle the correlated density is the unrelaxed MP2 one.'''
    mf = mf_cache('oh')
    m = pycmf.OBMP2(mf, max_cycle=1)
    m.verbose = 0
    m.kernel()
    ref = numpy.array(mp.UMP2(mf).run().make_rdm1(ao_repr=True))
    assert abs(m.make_rdm1(correlated=True) - ref).max() < 1e-9
    assert abs(m.make_rdm1() - mf.make_rdm1()).max() < 1e-9

    mf = mf_cache('h2o', 'rhf')
    m = pycmf.OBMP2(mf, max_cycle=1)
    m.verbose = 0
    m.kernel()
    ref = mp.MP2(mf).run().make_rdm1(ao_repr=True)
    assert abs(m.make_rdm1(correlated=True) - ref).max() < 1e-9


def test_density_traces(mf_cache):
    mf = mf_cache('oh')
    s = mf.get_ovlp()
    for bch_order in (2, 1):
        m = pycmf.OBMP2(mf, bch_order=bch_order).run()
        for correlated in (False, True):
            dma, dmb = m.make_rdm1(correlated=correlated)
            assert abs(numpy.trace(dma.dot(s)) - 5) < 1e-9
            assert abs(numpy.trace(dmb.dot(s)) - 4) < 1e-9
            dm_mo = m.make_rdm1(ao_repr=False, correlated=correlated)
            assert abs(dm_mo[0] - dm_mo[0].T).max() < 1e-12


def test_spin_square(mf_cache):
    '''Reference values printed by the original implementation.'''
    mf = mf_cache('oh')
    m = pycmf.OBMP2(mf, conv_tol=1e-10).run()
    assert abs(m.spin_square()[0] - 0.752525) < 2e-6
    assert abs(m.spin_square(correlated=True)[0] - 0.75163026) < 2e-7
    assert m.spin_square()[0] < mf.spin_square()[0]
    assert pycmf.OBMP2(mf_cache('h2o', 'rhf')).spin_square() == (0., 1.)


def test_hyperfine_coupling(mf_cache):
    '''Fermi-contact couplings (MHz) of O and H in OH; original implementation.'''
    m = pycmf.OBMP2(mf_cache('oh'), conv_tol=1e-10).run()
    assert abs(m.hfcc() - numpy.array([-121.869926, -106.324312])).max() < 1e-3
    assert m.hfcc(atoms=[1], correlated=True).shape == (1,)


def test_dipole_and_koopmans(mf_cache):
    m = pycmf.OBMP2(mf_cache('h2o', 'rhf'), conv_tol=1e-10).run()
    dip = m.dip_moment(verbose=0)
    assert abs(dip[:2]).max() < 1e-8 and 2.0 < abs(dip[2]) < 3.0
    assert abs(m.ip_koopmans() + m.mo_energy[4]) < 1e-12
    assert abs(m.ea_koopmans() + m.mo_energy[5]) < 1e-12
    mu = pycmf.OBMP2(mf_cache('h2o', 'uhf'), conv_tol=1e-10).run()
    assert abs(mu.ip_koopmans()[0] - m.ip_koopmans()) < 1e-6
    assert abs(numpy.linalg.norm(mu.dip_moment(verbose=0)) - numpy.linalg.norm(dip)) < 1e-5
