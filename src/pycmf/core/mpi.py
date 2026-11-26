# Copyright 2021-2026 The pyCMF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Optional MPI support (mpi4py).

The parallel parts of pyCMF follow the SPMD model: the same script runs on
every rank, rank 0 owns the mean-field part and broadcasts it, the loops over
k-points are distributed and the results are summed with ``allreduce``.
Without mpi4py, or with a single process, every function is a no-op.

MPI is switched off by setting the environment variable ``PYCMF_NO_MPI``.
'''

import os
import numpy

comm = None
rank = 0
size = 1
if not os.environ.get('PYCMF_NO_MPI'):
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if size == 1:
            comm = None
    except ImportError:
        pass


def split(tasks):
    '''Tasks of this rank (round robin).'''
    return list(tasks)[rank::size]


def bcast(obj):
    '''Broadcast an object from rank 0.'''
    if comm is None:
        return obj
    return comm.bcast(obj if rank == 0 else None, root=0)


def root_call(fn, *args, **kwargs):
    '''Evaluate ``fn`` on rank 0 only and broadcast the result.'''
    if comm is None:
        return fn(*args, **kwargs)
    return comm.bcast(fn(*args, **kwargs) if rank == 0 else None, root=0)


def allreduce(arrays):
    '''Sum a list of arrays (or scalars) of identical layout over all ranks.

    Returns a list with the same shapes; the input is not modified.
    '''
    if comm is None:
        return arrays
    arrays = [numpy.asarray(a, dtype=numpy.complex128) for a in arrays]
    buf = numpy.concatenate([a.ravel() for a in arrays]) if arrays else numpy.zeros(0, complex)
    out = numpy.empty_like(buf)
    comm.Allreduce(buf, out, op=MPI.SUM)
    res, p = [], 0
    for a in arrays:
        res.append(out[p:p + a.size].reshape(a.shape))
        p += a.size
    return res
