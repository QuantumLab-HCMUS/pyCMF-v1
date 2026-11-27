#!/usr/bin/env python
'''Orbital-optimized double hybrid (OO-DH, experimental).

The double-hybrid energy E_KS[hybrid part] + a_c E_2 is minimized with
respect to the orbitals.  Started from a Mole, the Kohn-Sham equations of the
hybrid part are solved first, so e_dh is the usual (non-optimized) double
hybrid and e_tot - e_dh is the lowering by the orbital optimization.

Reference (ORCA 4.1.0, RI-B2PLYP and RI-B2PLYP with %mp2 OrbOpt true,
cc-pVDZ, cc-pVDZ/C, no frozen core):
    E(B2PLYP) = -76.353769929, E(OO-B2PLYP) = -76.353874683,
    lowering -0.104754 mEh.
'''

from pyscf import gto
import pycmf

mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
            basis='cc-pvdz', verbose=0)

print('%-10s %18s %18s %14s' % ('xc', 'E(DH)', 'E(OO-DH)', 'lowering (mEh)'))
for xc in ('B2PLYP', 'PBE0-DH', 'B2GP-PLYP'):
    m = pycmf.OODH(mol, xc=xc, conv_tol=1e-10).density_fit('cc-pvdz-ri').run()
    print('%-10s %18.10f %18.10f %14.6f' % (xc, m.e_dh, m.e_tot, 1e3 * (m.e_tot - m.e_dh)))

# OO-MP2 is the limit ax = ac = 1 without semilocal parts
m = pycmf.OODH(mol, xc='HF', ac=1., conv_tol=1e-10).density_fit('cc-pvdz-ri').run()
print('%-10s %18.10f %18.10f %14.6f' % ('OO-MP2', m.e_dh, m.e_tot, 1e3 * (m.e_tot - m.e_dh)))

# relaxed density: D + a_c gamma (no orbital response, the orbitals are optimal)
m = pycmf.OODH(mol, conv_tol=1e-10).run()
print('dipole (Debye), determinant:', m.dip_moment(verbose=0))
print('dipole (Debye), relaxed    :', m.dip_moment(correlated=True, verbose=0))
