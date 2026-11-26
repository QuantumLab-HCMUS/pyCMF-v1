#!/usr/bin/env python
'''
Wall time per OBMP2 cycle.

    python timing.py [basis] [n_threads]

The time of one cycle is (t(4 cycles) - t(1 cycle)) / 3, which removes the
set-up cost (AO integrals, density-fitting tensors).  The molecule is benzene.
'''

import sys
import time
from pyscf import gto, scf, lib
import pycmf

basis = sys.argv[1] if len(sys.argv) > 1 else 'cc-pvdz'
if len(sys.argv) > 2:
    lib.num_threads(int(sys.argv[2]))

ATOM = '''C 0 1.3970 0; C 1.2098 0.6985 0; C 1.2098 -0.6985 0
          C 0 -1.3970 0; C -1.2098 -0.6985 0; C -1.2098 0.6985 0
          H 0 2.4810 0; H 2.1486 1.2405 0; H 2.1486 -1.2405 0
          H 0 -2.4810 0; H -2.1486 -1.2405 0; H -2.1486 1.2405 0'''
mol = gto.M(atom=ATOM, basis=basis, verbose=0)


def run(mf, ncycle, **kwargs):
    mp = pycmf.OBMP2(mf, max_cycle=ncycle, conv_tol=1e-14, **kwargs)
    mp.verbose = 0
    t0 = time.time()
    mp.kernel()
    return time.time() - t0


def per_cycle(mf, **kwargs):
    return (run(mf, 4, **kwargs) - run(mf, 1, **kwargs)) / 3


print('benzene/%s: %d basis functions, %d occupied orbitals, %d threads'
      % (basis, mol.nao, mol.nelectron // 2, lib.num_threads()))
print('%-28s %10s' % ('calculation', 's / cycle'))
for label, make in (('restricted', scf.RHF), ('unrestricted', scf.UHF)):
    mf = make(mol).run()
    mf_df = make(mol).density_fit().run()
    print('%-28s %10.2f' % (label + ', exact integrals', per_cycle(mf)))
    print('%-28s %10.2f' % (label + ', density fitting', per_cycle(mf_df)))
    print('%-28s %10.2f' % (label + ', DF, O2BMP2', per_cycle(mf_df, c_ss=0., c_os=1.2)))
