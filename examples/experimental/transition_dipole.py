#!/usr/bin/env python
'''Transition dipole moment and oscillator strength from state-specific
OBMP2 / O2BMP2 (experimental).

1. Converge the HF ground state; excite one beta electron 4 -> 5 with MOM-UHF.
2. Optimize both determinants with the correlated Fock operator.
3. Transition density of the two non-orthogonal determinants (generalized
   Slater-Condon rules), symmetric orthogonalization, mu and f = 2/3 dE |mu|^2.
   singlet=True multiplies mu by sqrt(2) (alpha and beta branches assumed
   equivalent and orthogonal); NOCI (noci.py) avoids that assumption.
'''

import numpy
from pyscf import gto, scf
from pyscf.data import nist
import pycmf
from pycmf.mol import excited
from pycmf.properties import transition

mol = gto.M(atom='''O 0.0000000000 0.0000000000 -0.0699025580
                    H 0.0000000000 0.7575324071 0.5184349453
                    H 0.0000000000 -0.7575324071 0.5184349453''',
            basis='cc-pvdz', verbose=0)
mf = scf.UHF(mol)
mf.conv_tol = 1e-10
mf.kernel()
mf_es = excited.mom_excitation(mf, hole=4, particle=5, spin=1)

print('%-14s %10s %12s %10s %10s' % ('method', 'dE (eV)', '|mu| (e a0)', 'f', '<S^2> ES'))
for name, make in (('OBMP2', pycmf.OBMP2),
                   ('O2BMP2 (1.2)', lambda mf, **kw: pycmf.O2BMP2(mf, c_os=1.2, **kw))):
    gs = make(mf, conv_tol=1e-10).run()
    es = make(mf_es, occupation='imom', conv_tol=1e-10).run()
    res = transition.transition_dipole(gs, es, singlet=True)
    print('%-14s %10.5f %12.6f %10.6f %10.4f' % (name, res.e_exc * nist.HARTREE2EV,
          numpy.linalg.norm(res.dipole), res.f, res.spin_square[1]))
