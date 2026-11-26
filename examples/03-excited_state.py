#!/usr/bin/env python
'''
State-specific excited state with the maximum overlap method.

1. Converge a non-Aufbau HF determinant with PySCF's MOM.
2. Re-optimise it with the correlated Fock operator, keeping the occupation
   by maximum overlap (occupation='imom' follows the initial determinant,
   'mom' the previous cycle).
The excitation energy is the difference of two total energies.
'''

from pyscf import gto, scf
import pycmf

mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
            basis='aug-cc-pvdz', verbose=0)
mf_gs = scf.UHF(mol).run()

# promote one alpha electron HOMO -> LUMO and converge the determinant
occ = mf_gs.mo_occ.copy()
homo = mol.nelec[0] - 1
occ[0][homo], occ[0][homo + 1] = 0, 1
mf_ex = scf.addons.mom_occ(scf.UHF(mol), mf_gs.mo_coeff, occ)
mf_ex.kernel(mf_ex.make_rdm1(mf_gs.mo_coeff, occ))

EV = 27.2114
print('%-16s %12s %8s' % ('method', 'omega (eV)', '<S^2>'))
print('%-16s %12.3f %8.3f' % ('MOM-HF', (mf_ex.e_tot - mf_gs.e_tot) * EV,
                              mf_ex.spin_square()[0]))
for name, make in (('OBMP2', lambda mf, **kw: pycmf.OBMP2(mf, **kw)),
                   ('O2BMP2 (1.2)', lambda mf, **kw: pycmf.O2BMP2(mf, c_os=1.2, **kw))):
    gs = make(mf_gs).run()
    ex = make(mf_ex, occupation='imom').run()
    print('%-16s %12.3f %8.3f   converged: %s'
          % (name, (ex.e_tot - gs.e_tot) * EV, ex.spin_square()[0], ex.converged))
print('The excited determinant is a 50:50 mixture of singlet and triplet (<S^2> ~ 1).')
