#!/usr/bin/env python
'''RO-OBMP2 (experimental): spin-pure high-spin open-shell calculation.'''

from pyscf import gto, scf
import pycmf
from pycmf.mol.roobmp2 import ROOBMP2

mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1, verbose=0)
mf = scf.ROHF(mol).run()
ro = ROOBMP2(mf).run()
uo = pycmf.OBMP2(scf.UHF(mol).run()).run()
print('E(RO-OBMP2) = %.8f   <S^2> = %.6f' % (ro.e_tot, ro.spin_square()[0]))
print('E(U-OBMP2)  = %.8f   <S^2> = %.6f' % (uo.e_tot, uo.spin_square()[0]))
