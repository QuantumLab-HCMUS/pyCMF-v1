#!/usr/bin/env python
'''k-point OBMP2 with MPI (experimental).

    OMP_NUM_THREADS=1 mpiexec -n 4 python kobmp2_mpi.py

The same script runs on every process.  The mean-field calculation is done by
process 0, which saves the three-index integrals to a file; the other
processes read that file and receive the orbitals.  KOBMP2 then distributes
the loop over k-points of the correlation potential.  Set the number of
threads so that processes times threads does not exceed the number of cores.
'''

import os
import tempfile

from pyscf.pbc import gto, scf
from pycmf.core import mpi
from pycmf.pbc import KOBMP2

a0 = 3.5668
cell = gto.M(atom='C 0 0 0; C %f %f %f' % (a0 / 4, a0 / 4, a0 / 4),
             a=[[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]],
             basis='gth-szv', pseudo='gth-pade', verbose=0)
kpts = cell.make_kpts([3, 3, 3])
cderi = mpi.bcast(os.path.join(tempfile.gettempdir(), 'kobmp2_mpi_cderi_%d.h5' % os.getpid()))

mf = scf.KRHF(cell, kpts).density_fit()
if mpi.rank == 0:
    mf.with_df._cderi_to_save = cderi
    mf.kernel()
state = mpi.bcast((mf.mo_energy, mf.mo_coeff, mf.mo_occ, mf.e_tot) if mpi.rank == 0 else None)
if mpi.rank > 0:
    mf.with_df._cderi = cderi
    mf.mo_energy, mf.mo_coeff, mf.mo_occ, mf.e_tot = state

mp = KOBMP2(mf)
mp.kernel()
if mpi.rank == 0:
    print('MPI processes = %d' % mpi.size)
    print('E(KRHF)   = %.8f' % mf.e_tot)
    print('E(KOBMP2) = %.8f' % mp.e_tot)
    os.remove(cderi)
