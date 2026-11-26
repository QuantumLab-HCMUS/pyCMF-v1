#!/usr/bin/env python
'''Projection-based embedding (experimental): one water molecule of a water
dimer at the OBDH level in a PBE environment.

The two levels have their own functionals: PBE (``xc_env``) for the
environment and the embedding potential, the double hybrid ``xc`` (B2PLYP by
default) for the subsystem.  A converged Kohn-Sham calculation of the full
system is used as the low level.'''

from pyscf import gto, dft
from pycmf.embed import EmbeddedOBDH, EmbeddedOBMP2

mol = gto.M(atom='''O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692
                    O 0 0 3.0173; H 0 0.7572 3.6038; H 0 -0.7572 3.6038''',
            basis='cc-pvdz', verbose=0)
ks = dft.UKS(mol, xc='pbe').density_fit().run()
print('E(PBE, full system) = %.8f' % ks.e_tot)

for use_cl in (False, True):
    emb = EmbeddedOBDH(ks, active_atoms=[0, 1, 2], xc='B2PLYP', use_cl=use_cl, n_shells=1)
    emb.kernel()
    print('B2PLYP-OBDH in %s, CL truncation: %-5s  E = %.8f   virtuals kept: %d'
          % (emb.get_xc_env(), use_cl, emb.e_tot,
             emb.mo_coeff[0].shape[1] - int(emb.mo_occ[0].sum())))
print('dipole moment of the full system (D) = %.4f' % emb.dip_moment(verbose=0)[2])

emb = EmbeddedOBMP2(ks, active_atoms=[0, 1, 2])
print('OBMP2 in PBE                             E = %.8f' % emb.kernel())

# DFT-in-DFT with the same functional on both levels returns the Kohn-Sham energy
emb = EmbeddedOBDH(ks, active_atoms=[0, 1, 2], xc='pbe', ac=0.)
print('PBE in PBE - E(PBE)                        = %.1e' % abs(emb.kernel() - ks.e_tot))
