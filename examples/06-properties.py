#!/usr/bin/env python
'''
First-order properties: densities, dipole moments, <S^2> and isotropic
hyperfine couplings of the CN radical.

Two densities are available.  correlated=False gives the density of the
OBMP2 determinant ("HF-like"); correlated=True adds the unrelaxed
second-order correction from the converged amplitudes ("MP2-like").
'''

import numpy
from pyscf import gto, scf
import pycmf

mol = gto.M(atom='C 0 0 0; N 0 0 1.1718', basis='cc-pvdz', spin=1, verbose=0)
mf = scf.UHF(mol).run()
mp = pycmf.OBMP2(mf).run()

print('<S^2>: UHF %.4f | OBMP2 determinant %.4f | with amplitudes %.4f'
      % (mf.spin_square()[0], mp.spin_square()[0], mp.spin_square(correlated=True)[0]))
for label, corr in (('HF-like ', False), ('MP2-like', True)):
    dip = numpy.linalg.norm(mp.dip_moment(correlated=corr, verbose=0))
    a_iso = mp.hfcc(correlated=corr)
    print('%s density: dipole %.4f D   A_iso(13C) %8.2f MHz   A_iso(14N) %8.2f MHz'
          % (label, dip, a_iso[0], a_iso[1]))

dma, dmb = mp.make_rdm1(correlated=True)          # AO basis
occ = numpy.linalg.eigvalsh(mp.make_rdm1(ao_repr=False, correlated=True)[0])
print('alpha natural occupations (largest 3 virtual): %s' % numpy.round(occ[:-7][-3:], 4))
print('Koopmans IP (eV): alpha %.3f  beta %.3f' % tuple(27.2114 * x for x in mp.ip_koopmans()))
