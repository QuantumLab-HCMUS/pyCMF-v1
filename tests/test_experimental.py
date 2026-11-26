'''Methods under development: only regression against the original code.'''

import warnings

import numpy
import pytest
from pyscf import dft, gto, scf

from pycmf.core.status import ExperimentalWarning

# RO-OBMP2, original implementation: (BCH2, BCH1)
REF_RO = {
    'oh': (-75.45169934628294, -75.44923834487469),
    'ch2': (-38.961458979964384, -38.96012797854658),
    'nh2': (-55.616924799431196, -55.61452640654551),
}


@pytest.mark.parametrize('name', ['oh', 'ch2', 'nh2'])
def test_roobmp2(mf_cache, name):
    from pycmf.mol.roobmp2 import ROOBMP2
    mf = mf_cache(name, 'rohf')
    with pytest.warns(ExperimentalWarning):
        m = ROOBMP2(mf)
    m.conv_tol = 1e-10
    m.max_cycle = 300
    assert abs(m.kernel() - REF_RO[name][0]) < 1e-8
    assert m.converged
    s = mf.mol.spin / 2
    assert abs(m.spin_square()[0] - s * (s + 1)) < 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m1 = ROOBMP2(mf).set(bch_order=1, conv_tol=1e-10, max_cycle=300)
    assert abs(m1.kernel() - REF_RO[name][1]) < 1e-8


# projection embedding of one water molecule in a water dimer, PBE
# environment; the original implementation used the environment functional
# for the subsystem as well, which is reproduced here with xc='pbe'
# 0.2 energies.  In 0.2 the functional of the environment was also the
# Kohn-Sham part of the embedded "OBDH"; this is reproduced with xc=xc_env.
REF_EMB = {
    'obdh': -152.61065287710954,
    'obdh_cl': -152.6106502832336,
    'obmp2': -152.40211686622075,
    'obmp2_cl': -152.4021158267036,
}
# B2PLYP-in-PBE with the two levels separated (1.0)
REF_EMB_B2PLYP_IN_PBE = -152.57285977387573


@pytest.fixture(scope='module')
def dimer():
    mol = gto.M(atom='''O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692
                        O 0 0 3.0173; H 0 0.7572 3.6038; H 0 -0.7572 3.6038''',
                basis='6-31g', verbose=0)
    return scf.UHF(mol).density_fit().run(conv_tol=1e-11)


@pytest.fixture(scope='module')
def dimer_ks(dimer):
    ks = dft.UKS(dimer.mol, xc='pbe').density_fit()
    ks.with_df = dimer.with_df
    return ks.run(conv_tol=1e-11)


@pytest.mark.parametrize('use_cl', [False, True])
def test_embedding(dimer, use_cl):
    from pycmf.embed import EmbeddedOBDH, EmbeddedOBMP2
    kw = dict(active_atoms=[0, 1, 2], xc_env='pbe', use_cl=use_cl, conv_tol=1e-9)
    tag = '_cl' if use_cl else ''
    with pytest.warns(ExperimentalWarning):
        m = EmbeddedOBDH(dimer, xc='pbe', ac=0.27, **kw)
    assert abs(m.kernel() - REF_EMB['obdh' + tag]) < 1e-7
    assert m.converged
    with pytest.warns(ExperimentalWarning):
        m = EmbeddedOBMP2(dimer, **kw)
    assert abs(m.kernel() - REF_EMB['obmp2' + tag]) < 1e-7


@pytest.mark.filterwarnings('ignore::pycmf.core.status.ExperimentalWarning')
def test_embedding_whole_system_is_the_high_level_method(dimer):
    import pycmf
    from pycmf.embed import EmbeddedOBDH, EmbeddedOBMP2
    atoms = list(range(dimer.mol.natm))
    m = EmbeddedOBDH(dimer, active_atoms=atoms, xc_env='pbe', conv_tol=1e-10)
    assert abs(m.kernel() - pycmf.OBDH(dimer, conv_tol=1e-10).kernel()) < 1e-8
    assert m.nelec_sub == dimer.mol.nelec
    m = EmbeddedOBMP2(dimer, active_atoms=atoms, xc_env='pbe', conv_tol=1e-10)
    assert abs(m.kernel() - pycmf.OBMP2(dimer, conv_tol=1e-10).kernel()) < 1e-8


@pytest.mark.filterwarnings('ignore::pycmf.core.status.ExperimentalWarning')
def test_embedding_dft_in_dft_is_exact(dimer_ks):
    from pycmf.embed import EmbeddedOBDH
    m = EmbeddedOBDH(dimer_ks, active_atoms=[0, 1, 2], xc='pbe', ac=0., conv_tol=1e-10)
    assert m.get_xc_env() == 'pbe'           # taken from the Kohn-Sham object
    assert abs(m.kernel() - dimer_ks.e_tot) < 1e-7
    assert abs(m.e_env - dimer_ks.e_tot) < 1e-12


@pytest.mark.filterwarnings('ignore::pycmf.core.status.ExperimentalWarning')
def test_embedding_double_hybrid_in_dft(dimer, dimer_ks):
    from pycmf.embed import EmbeddedOBDH
    # default: B2PLYP for the subsystem, PBE for the environment
    m = EmbeddedOBDH(dimer, active_atoms=[0, 1, 2], conv_tol=1e-9)
    assert abs(m.kernel() - REF_EMB_B2PLYP_IN_PBE) < 1e-7
    assert m.get_xc() == '0.53*HF + 0.47*B88, 0.73*LYP' and m.get_xc_env() == 'pbe'
    # the same from a converged Kohn-Sham calculation of the full system
    m = EmbeddedOBDH(dimer_ks, active_atoms=[0, 1, 2], xc='B2PLYP', conv_tol=1e-9)
    assert abs(m.kernel() - REF_EMB_B2PLYP_IN_PBE) < 1e-7
    # density and dipole moment of the full system
    dm = m.make_rdm1()
    assert abs(numpy.einsum('sij,ji->', dm, dimer.get_ovlp()) - dimer.mol.nelectron) < 1e-8
    assert abs(numpy.einsum('sij,ji->', m.make_rdm1(with_env=False), dimer.get_ovlp()) - 10) < 1e-8
    assert abs(m.dip_moment(verbose=0)[2]) < 0.5


def test_experimental_methods_are_not_exported():
    import pycmf
    for name in ('ROOBMP2', 'KOBMP2', 'KUOBMP2', 'KOBDF', 'EmbeddedOBDH'):
        assert not hasattr(pycmf, name)
