'''Non-orthogonal CI in a basis of state-specific OBMP2 determinants.'''

import functools

import numpy
import pytest
from pyscf import gto, scf

import pycmf
from pycmf.mol import excited, noci
from pycmf.properties import transition
from fci_reference import FCISpace

pytestmark = pytest.mark.filterwarnings('ignore::pycmf.core.status.ExperimentalWarning')

H2O_CC3 = '''O 0.0000000000 0.0000000000 -0.0699025580
H 0.0000000000 0.7575324071 0.5184349453
H 0.0000000000 -0.7575324071 0.5184349453'''

# calc_noci_ob_o2b.py with the 0.2 library, OBMP2/cc-pVDZ, 2 channels
# (4->5, 4->6; alpha and beta), bare Hamiltonian
E_BARE_REF = [-76.02593706410548, -75.7788533659335, -75.75342569279312,
              -75.6956601427206, -75.6814161710584]
F_BARE_REF = 0.020275393542891353      # root 2, the bright singlet
MU_BARE_REF = 0.33407037242274157


@pytest.fixture(scope='module')
def small():
    '''Distorted water, STO-3G: small enough for the full CI reference.'''
    mol = gto.M(atom='O 0 0 0.05; H 0.1 0.78 -0.47; H -0.05 -0.74 -0.45',
                basis='sto-3g', verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1e-12)
    labels, states = noci.make_basis(mf, [(4, 5), (3, 6)])
    return mol, mf, labels, states


def determinants(mol, mf, states):
    '''OBMP2 determinants, an MOM-UHF one and an unrelaxed single excitation
    (singular orbital overlap with the HF determinant).'''
    dets = [noci.Determinant.from_state(st, 'D%d' % i) for i, st in enumerate(states)]
    dets.append(noci.Determinant.from_state(excited.mom_excitation(mf, 2, 6, 1), 'mom'))
    ca, cb = mf.mo_coeff
    dets.append(noci.Determinant((ca[:, [0, 1, 2, 3, 5]], cb[:, :5]), 0., 'swap'))
    dets.append(noci.Determinant((ca[:, :5], cb[:, :5]), mf.e_tot, 'hf'))
    return dets


@pytest.mark.parametrize('direct', [False, True])
def test_matrices_against_fci(small, direct):
    '''S, H, dipole and S^2 equal the full CI expectation values.'''
    mol, mf, labels, states = small
    dets = determinants(mol, mf, states)
    m = noci.NOCI(dets, mol=mol)
    m.direct = direct
    S, H, D, S2 = m.build()
    space = FCISpace(mol, mf.mo_coeff[0], mol.nelec)
    S0, H0, D0, S20 = space.matrices([d.occ_coeff for d in dets])
    assert abs(S - S0).max() < 1e-10
    assert abs(H - H0).max() < 1e-9
    assert abs(D - D0).max() < 1e-9
    assert abs(S2 - S20).max() < 1e-9
    assert abs(S[-1, -2]) < 1e-12  # the singular pair is exercised
    assert abs(H[-1, -2]) < 1e-9       # Brillouin: <HF|H|i->a> = f_ia = 0
    assert abs(D[:, -1, -2]).max() > 1e-2


def test_diagonals(small):
    '''Bare diagonal = HF energy of the OBMP2 determinant; the two effective
    Hamiltonians have the OBMP2 energies on the diagonal.'''
    mol, mf, labels, states = small
    m = noci.NOCI(states, labels)
    m.build()
    assert abs(numpy.diag(m.h_bare) - [st.e_ref for st in states]).max() < 1e-9
    e = [st.e_tot for st in states]
    for model in ('overlap_symmetric', 'state_specific_fock'):
        assert abs(numpy.diag(m.get_hamiltonian(model)) - e).max() < 1e-9


def test_effective_constant(small):
    '''c_I = E_I - sum_occ F_I equals E_nuc - 1/2 sum_occ v_HF + c0 of the
    correlation potential.'''
    mol, mf, labels, states = small
    for st in states:
        det = noci.Determinant.from_state(st)
        im = st.get_intermediates()
        vhf = st._scf.get_veff(mol, numpy.asarray(im.dm_chan))
        e_hf2 = sum(numpy.einsum('pq,qp->', v, d) for v, d in zip(vhf, im.dm_chan))
        ref = mol.energy_nuc() - .5 * e_hf2 + im.c0
        assert abs(det.scalar - ref) < 1e-9


def test_effective_symmetric(small):
    mol, mf, labels, states = small
    dets = [noci.Determinant.from_state(st) for st in states]
    s1e = mol.intor('int1e_ovlp')
    for a in dets:
        for b in dets:
            assert abs(noci.effective_pair(a, b, s1e)[1]
                       - noci.effective_pair(b, a, s1e)[1]) < 1e-10


def test_single_determinant(small):
    mol, mf, labels, states = small
    assert abs(noci.NOCI(states[:1]).kernel('state_specific_fock')[0] - states[0].e_tot) < 1e-10
    assert abs(noci.NOCI(states[:1]).kernel('bare')[0] - states[0].e_ref) < 1e-10


def test_linear_dependency(small):
    mol, mf, labels, states = small
    m = noci.NOCI(states + states[1:2], labels + ['copy'])
    e = m.kernel('bare')
    ref = noci.NOCI(states, labels).kernel('bare')
    assert m.nlindep == 1
    assert abs(e - ref).max() < 1e-8
    assert abs(m.ci.T @ m.s @ m.ci - numpy.eye(len(e))).max() < 1e-8


def test_channel_weights(small):
    mol, mf, labels, states = small
    m = noci.NOCI(states, labels)
    m.kernel('bare')
    w = m.channel_weights()
    assert set(w) == {(4, 5), (3, 6)}
    for v in w.values():
        assert (v > -1e-12).all() and (v < 1 + 1e-9).all()
    assert m.reference_weight()[0] > .9


def test_input_checks(small):
    mol, mf, labels, states = small
    cation = scf.UHF(gto.M(atom=mol.atom, basis='sto-3g', charge=1, spin=1, verbose=0)).run()
    with pytest.raises(ValueError):
        noci.NOCI([states[0], cation], mol=mol)
    with pytest.raises(ValueError):
        noci.NOCI(states, labels).kernel('unknown')
    with pytest.raises(ValueError):
        noci.NOCI(states, labels[:1])
    with pytest.raises(RuntimeError):
        noci.NOCI([noci.Determinant(transition.occupied_orbitals(mf), mf.e_tot)],
                  mol=mol).kernel('state_specific_fock')


def test_regression_h2o_bare():
    '''Reproduces calc_noci_ob_o2b.py (bare model) with the 0.2 library.'''
    mol = gto.M(atom=H2O_CC3, basis='cc-pvdz', verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    channels = excited.select_channels(mf, 2)
    assert [c[:2] for c in channels] == [(4, 5), (4, 6)]
    labels, states = noci.make_basis(mf, channels)
    assert labels == ['GS', 'h4_p5_a', 'h4_p5_b', 'h4_p6_a', 'h4_p6_b']
    m = noci.NOCI(states, labels)
    e = m.kernel('bare')
    assert abs(e - E_BARE_REF).max() < 1e-7
    assert abs(m.f[2] - F_BARE_REF) < 1e-7
    assert abs(numpy.linalg.norm(m.dipole[:, 2]) - MU_BARE_REF) < 1e-7
    assert abs(m.spin_square[1] - 2) < 1e-3 and abs(m.spin_square[2]) < .02
    # the state-specific Fock model keeps the alpha and beta branches degenerate
    e_ssf = m.kernel('state_specific_fock')
    assert abs(e_ssf[0] - states[0].e_tot) < 1e-6
    assert abs(e_ssf[1] - e_ssf[2]) < 1e-6


def test_o2bmp2_basis(small):
    '''make_basis passes the method through (O2BMP2 differs from OBMP2).'''
    mol, mf, labels, states = small
    lab2, st2 = noci.make_basis(mf, [(4, 5)], method=functools.partial(pycmf.O2BMP2, c_os=1.2))
    assert all(s.c_ss == 0 and s.c_os == 1.2 for s in st2)
    assert abs(st2[0].e_tot - states[0].e_tot) > 1e-4
