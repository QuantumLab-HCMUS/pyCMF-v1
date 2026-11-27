#!/usr/bin/env python
'''NOCI in a basis of state-specific OBMP2 determinants (experimental).

The ground state and, for the two lowest HF excitation channels, the alpha and
the beta excited determinant are optimized separately and combined by solving
H C = S C E.  With the bare Hamiltonian the alpha/beta pairs split into
triplets (<S^2> = 2) and singlets; only the singlets carry oscillator strength.
'''

import warnings

import numpy
from pyscf import gto, scf
from pyscf.data import nist
from pycmf.mol import excited, noci

warnings.simplefilter('ignore')
mol = gto.M(atom='''O 0.0000000000 0.0000000000 -0.0699025580
                    H 0.0000000000 0.7575324071 0.5184349453
                    H 0.0000000000 -0.7575324071 0.5184349453''',
            basis='cc-pvdz', verbose=0)
mf = scf.UHF(mol)
mf.conv_tol = 1e-10
mf.kernel()

channels = excited.select_channels(mf, 2)            # (4, 5) and (4, 6)
labels, states = noci.make_basis(mf, channels)       # GS + 4 determinants
m = noci.NOCI(states, labels)
for model in ('bare', 'overlap_symmetric'):
    m.kernel(model)
    print(model)
    print('%5s %16s %10s %10s %10s %8s' % ('root', 'E (Eh)', 'dE (eV)', '|mu|', 'f', '<S^2>'))
    for k in range(len(m.e)):
        print('%5d %16.8f %10.5f %10.6f %10.6f %8.4f' % (
            k, m.e[k], m.e_exc[k] * nist.HARTREE2EV, numpy.linalg.norm(m.dipole[:, k]),
            m.f[k], m.spin_square[k]))
