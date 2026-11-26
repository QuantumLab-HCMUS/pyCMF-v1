#!/usr/bin/env python
'''
One-body double hybrids: OBDH and O2BDH.

The Kohn-Sham Fock matrix of a double-hybrid functional is augmented with the
scaled OBMP2 correlation potential and diagonalised self-consistently.  The
functional is B2PLYP by default; other double hybrids are chosen by name, by
the four parameters, or by a PySCF functional string.  The published
parameters have not been re-optimised for the self-consistent scheme.
'''

from pyscf import gto, scf, dft
import pycmf

mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
            basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).density_fit().run()

dh = pycmf.OBDH(mf).run()
print('functional          : %s  + %.2f E_c(OBMP2)' % (dh.get_xc(), dh.ac))
print('E(OBDH)             = %.10f   converged: %s' % (dh.e_tot, dh.converged))
print('E(O2BDH, c_os=1.2)  = %.10f' % pycmf.O2BDH(mf, c_os=1.2).kernel())
print('dipole moment (D)   = %.4f' % abs(dh.dip_moment(verbose=0)[2]))

# double hybrids by name
for name in ('B2GP-PLYP', 'PBE0-DH', 'PBE-QIDH'):
    m = pycmf.OBDH(mf, xc=name).run()
    print('%-10s a_x = %.4f  a_c = %.4f  E = %.10f' % (name, m.ax, m.ac, m.e_tot))

# the four parameters, with Kohn-Sham starting orbitals
ks = dft.RKS(mol, xc='pbe0').density_fit().run()
dh2 = pycmf.OBDH(ks, ax=0.5, ac=0.25, xc_x='PBE', xc_c='PBE').run()
print('custom  %s : E = %.10f' % (dh2.get_xc(), dh2.e_tot))

mol_oh = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1, verbose=0)
dh3 = pycmf.OBDH(scf.UHF(mol_oh).density_fit().run()).run()
print('OH: E(UOBDH) = %.10f   <S^2> = %.4f' % (dh3.e_tot, dh3.spin_square()[0]))
