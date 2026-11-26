#!/usr/bin/env python
'''k-point one-body downfolding (experimental): diamond, minimal basis,
2x2x1 k-points, three valence and three conduction bands active.

The correlation with the orbitals outside the active bands is folded into the
one-body part of an active-space Hamiltonian; its two-electron integrals come
from the three-index tensors of the KOBMP2 calculation.
'''

import numpy
from pyscf.pbc import gto, scf
from pycmf.pbc import KOBMP2

a0 = 3.5668
cell = gto.M(atom='C 0 0 0; C %f %f %f' % (a0 / 4, a0 / 4, a0 / 4),
             a=[[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]],
             basis='gth-szv', pseudo='gth-pade', verbose=0)
mf = scf.KRHF(cell, cell.make_kpts([2, 2, 1]), exxdiv=None).density_fit().run(conv_tol=1e-10)
mp = KOBMP2(mf).run()
print('E(KRHF) = %.8f   E(KOBMP2) = %.8f   gap of the KOBMP2 bands = %.5f'
      % (mf.e_tot, mp.e_tot, mp.band_gap()))

ham = mp.downfold(ncas=6, nelecas=6)          # same as pycmf.pbc.KOBDF(mp, 6, 6)
h1, e_core = ham.get_h1eff()
print('h1eff: %s   E_core = %.8f' % (h1.shape, e_core))
print('sum rule: <H_act> - [E(KOBMP2) - E_c(T_act)] = %.1e'
      % abs(ham.energy_tot() - (mp.e_tot - ham.e_corr_act)))

# mean field and second-order quasiparticle energies of the model
print('E(HF of H_act) = %.8f   gap = %.5f' % (ham.scf(), ham.band_gap()))
ham.self_energy(nocc=1, nvir=1)
for scheme in ('static', 'linear', 'dynamic'):
    print('second-order gap of H_act, %-7s = %.5f' % (scheme, ham.quasiparticle_gap(scheme)[1]))

# integrals for other solvers
print('h2eff block (k1, k2, k3) = (1, 2, 3): norm %.6f' % numpy.linalg.norm(ham.get_h2eff(1, 2, 3)))

# many-body solvers work on the real Hamiltonian of the supercell (4 unit cells)
from pyscf import cc
h1s, h2s, e0, nelec = ham.supercell()
mf_sc = ham.as_scf().run()
e_cc = cc.CCSD(mf_sc).run().e_tot / 4
print('supercell: %d orbitals, %d electrons; E(HF) = %.8f, E(CCSD) = %.8f per cell'
      % (h1s.shape[0], nelec, mf_sc.e_tot / 4, e_cc))
