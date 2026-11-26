#!/usr/bin/env python
'''k-point OBMP2 and O2BMP2 (experimental): diamond, minimal basis, 2x2x2 k-points.

The correlation potential is built from Gaussian density-fitted integrals; the
k-point symmetry of the crystal reduces the eight k-points to three.
'''

from pyscf.pbc import gto, scf
from pycmf.pbc import KOBMP2, KO2BMP2

a0 = 3.5668
cell = gto.M(atom='C 0 0 0; C %f %f %f' % (a0 / 4, a0 / 4, a0 / 4),
             a=[[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]],
             basis='gth-szv', pseudo='gth-pade', verbose=0)
mf = scf.KRHF(cell, cell.make_kpts([2, 2, 2])).density_fit().run(conv_tol=1e-10)

mp = KOBMP2(mf).run()
gap_hf = min(e[o == 0].min() for e, o in zip(mf.mo_energy, mf.mo_occ)) \
    - max(e[o > 0].max() for e, o in zip(mf.mo_energy, mf.mo_occ))
print('E(KRHF)    = %.8f   gap = %.5f' % (mf.e_tot, gap_hf))
print('E(KOBMP2)  = %.8f   gap = %.5f' % (mp.e_tot, mp.band_gap()))

# the same calculation without k-point symmetry
ref = KOBMP2(mf, kpt_symmetry=False).run()
print('without k-point symmetry: %d k-points instead of %d, same energy: %s'
      % (ref._sym.nibz, mp._sym.nibz, abs(ref.e_tot - mp.e_tot) < 1e-7))

# opposite-spin scaled variant (c_os has no default)
o2 = KO2BMP2(mf, c_os=1.2).run()
print('E(KO2BMP2, c_os = 1.2) = %.8f   gap = %.5f' % (o2.e_tot, o2.band_gap()))

# Frozen bands: KOBMP2(mf, frozen=n).  The frozen set must not cut a degenerate
# level; in diamond the two lowest valence bands are degenerate at the X point,
# so that frozen=1 is not a valid choice for this mesh (a warning is issued).
