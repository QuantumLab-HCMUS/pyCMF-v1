#!/usr/bin/env python
'''
Spin-opposite-scaled OBMP2 (O2BMP2): singlet-triplet gap of methylene.

Only opposite-spin amplitudes are kept and scaled by c_os.  The scaling acts
on the correlation potential, so it changes the orbitals and not only the
energy.  c_os must be chosen by the user; 1.2 was used for singlet-triplet
gaps in J. Phys. Chem. A 128, 1543 (2024).
'''

from pyscf import gto, scf
import pycmf

HARTREE2KCAL = 627.5095
basis = 'cc-pvtz'
triplet = gto.M(atom='C 0 0 0; H 0 0.9972 0.3987; H 0 -0.9972 0.3987',
                basis=basis, spin=2, verbose=0)
singlet = gto.M(atom='C 0 0 0; H 0 0.8623 0.6845; H 0 -0.8623 0.6845',
                basis=basis, spin=0, verbose=0)

mf_t = scf.UHF(triplet).density_fit().run()
mf_s = scf.RHF(singlet).density_fit().run()
print('%-18s %10s' % ('method', 'E(S)-E(T) kcal/mol'))
print('%-18s %10.2f' % ('HF', (mf_s.e_tot - mf_t.e_tot) * HARTREE2KCAL))
gap = pycmf.OBMP2(mf_s).kernel() - pycmf.OBMP2(mf_t).kernel()
print('%-18s %10.2f' % ('OBMP2', gap * HARTREE2KCAL))
for c_os in (1.1, 1.2, 1.3):
    gap = pycmf.O2BMP2(mf_s, c_os=c_os).kernel() - pycmf.O2BMP2(mf_t, c_os=c_os).kernel()
    print('%-18s %10.2f' % ('O2BMP2 c_os=%.1f' % c_os, gap * HARTREE2KCAL))
