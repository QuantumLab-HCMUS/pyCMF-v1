#!/usr/bin/env python
'''
OBMP2 ground-state calculation.

The reference decides the flavour: RHF gives restricted OBMP2, UHF or ROHF
give unrestricted OBMP2.  One cycle reproduces conventional MP2; the
converged result is the self-consistent one.
'''

from pyscf import gto, scf, mp
import pycmf

mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
            basis='cc-pvdz', verbose=3)

# closed shell, exact integrals
mf = scf.RHF(mol).run()
ob = pycmf.OBMP2(mf).run()
print('E(RHF)    = %.10f' % mf.e_tot)
print('E(MP2)    = %.10f' % mp.MP2(mf).run().e_tot)
print('E(OBMP2)  = %.10f   converged: %s' % (ob.e_tot, ob.converged))
print('Koopmans IP (eV) = %.3f' % (ob.ip_koopmans() * 27.2114))

# density fitting: reuse the DF object of a DF-HF calculation ...
mf_df = scf.RHF(mol).density_fit().run()
print('E(DF-OBMP2), JK-fit basis   = %.10f' % pycmf.OBMP2(mf_df).kernel())
# ... or fit only the correlation part with an RI-C auxiliary basis
print('E(DF-OBMP2), RI-C basis     = %.10f'
      % pycmf.OBMP2(mf).density_fit(auxbasis='cc-pvdz-ri').kernel())

# open shell
mol_oh = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1, verbose=3)
mf_oh = scf.UHF(mol_oh).run()
ob_oh = pycmf.OBMP2(mf_oh).run()
print('OH:  E(UOBMP2) = %.10f' % ob_oh.e_tot)
print('     <S^2>: UHF %.4f   OBMP2 %.4f' % (mf_oh.spin_square()[0], ob_oh.spin_square()[0]))
