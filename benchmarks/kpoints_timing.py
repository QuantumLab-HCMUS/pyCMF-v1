#!/usr/bin/env python
'''
Timing of k-point OBMP2 for diamond (two atoms per cell, GTH pseudopotential,
Gaussian density fitting).

    python kpoints_timing.py gth-szv 3x3x3          # full mesh and symmetry
    python kpoints_timing.py gth-dzvp 2x2x2 conv    # also cycles to convergence

Prints the time of the first cycle (with the one-time setup), the time per
further cycle, and the peak memory of the process.  The numbers quoted in
docs/kpoints.md were obtained with this script on two threads.
'''

import io
import resource
import sys
import time
import warnings

from pyscf.pbc import gto, scf
from pycmf.pbc import KOBMP2

warnings.filterwarnings('ignore')
A0 = 3.5668


def diamond(basis):
    return gto.M(atom='C 0 0 0; C %f %f %f' % (A0 / 4, A0 / 4, A0 / 4),
                 a=[[0, A0 / 2, A0 / 2], [A0 / 2, 0, A0 / 2], [A0 / 2, A0 / 2, 0]],
                 basis=basis, pseudo='gth-pade', verbose=0)


def timed(mf, mode, ncyc, **attrs):
    mp = KOBMP2(mf, kpt_symmetry=mode)
    mp.max_cycle = ncyc
    mp.__dict__.update(attrs)
    t0 = time.time()
    mp.kernel()
    return mp, time.time() - t0


if __name__ == '__main__':
    basis = sys.argv[1] if len(sys.argv) > 1 else 'gth-szv'
    mesh = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '2x2x2').split('x')]
    cell = diamond(basis)
    t0 = time.time()
    mf = scf.KRHF(cell, cell.make_kpts(mesh)).density_fit().run(conv_tol=1e-9)
    print('%s %s: %d AOs, %d auxiliary functions, KRHF %.0f s'
          % (basis, mesh, cell.nao, mf.with_df.get_naoaux(), time.time() - t0))
    for mode in (False, True):
        m1, t1 = timed(mf, mode, 1)
        m3, t3 = timed(mf, mode, 3)
        print('  kpt_symmetry=%-5s %2d of %2d k-points: first cycle %.1f s, per cycle %.2f s'
              % (mode, m1._sym.nibz, m1._sym.nkpts, t1, (t3 - t1) / 2))
    if 'conv' in sys.argv:
        for diis in (True, False):
            mp, t = timed(mf, True, 100, diis=diis, verbose=4, stdout=io.StringIO())
            print('  DIIS %-5s: %d cycles, %.1f s, E = %.10f, gap = %.5f'
                  % (diis, mp.stdout.getvalue().count('cycle='), t, mp.e_tot, mp.band_gap()))
    print('  peak memory %.2f GB' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
