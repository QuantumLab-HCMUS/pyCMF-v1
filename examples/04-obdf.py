#!/usr/bin/env python
'''
One-body downfolding (OBDF): dissociation of N2 in a (6e,6o) active space.

The OBMP2 correlation potential of the double excitations that leave the
active space is added to the one-electron integrals; the two-electron
integrals stay bare.  The active-space Hamiltonian is solved here by FCI and
written to a FCIDUMP file for other solvers.
'''

from pyscf import gto, scf, mcscf, cc
import pycmf

print('%6s %14s %14s %14s %14s' % ('R (A)', 'OBMP2', 'CASCI(6,6)', 'OBDF(6,6)', 'CCSD(T)'))
for r in (1.0, 1.1, 1.3, 1.6, 2.0):
    mol = gto.M(atom='N 0 0 0; N 0 0 %g' % r, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run()
    mp = pycmf.OBMP2(mf).run()
    ham = pycmf.OBDF(mp, ncas=6, nelecas=6)
    e_obdf = ham.kernel()
    e_cas = mcscf.CASCI(mf, 6, 6).kernel()[0]
    mycc = cc.CCSD(mf).run()
    e_ccsdt = mycc.e_tot + mycc.ccsd_t()
    print('%6.2f %14.6f %14.6f %14.6f %14.6f' % (r, mp.e_tot, e_cas, e_obdf, e_ccsdt))

# hand the Hamiltonian of the last geometry to another program
h1, e_core = ham.get_h1eff()
h2 = ham.get_h2eff()
ham.to_fcidump('n2_obdf.fcidump')
print('\nactive-space Hamiltonian: h1 %s, h2 %s, E_core = %.8f' % (h1.shape, h2.shape, e_core))
print('written to n2_obdf.fcidump')
