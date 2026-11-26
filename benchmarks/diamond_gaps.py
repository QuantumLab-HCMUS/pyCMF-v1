#!/usr/bin/env python
'''
Band gaps of diamond with KOBMP2 and its variants for one basis set and k-mesh.

    python diamond_gaps.py gth-dzvp 3x3x3 none      # exxdiv=None, as in the KOBMP2 paper
    python diamond_gaps.py gth-dzvp 4x4x4 ewald

Prints, in eV, the direct gap at Gamma and the gap from the valence band
maximum at Gamma to the lowest conduction band on the mesh for

    HF                        eigenvalues of KRHF
    MP2 on HF                 second-order quasiparticle energies with HF orbitals
    KOBMP2, SCS, SOS          eigenvalues of the correlated Fock matrix

and, for each correlated method, the quasiparticle gaps of pycmf.pbc.kqp:

    static     eigenvalue + other half of the static self-energy (IPEA)
    linear     the same with the renormalization factor Z
    dynamic    solution of the quasiparticle equation
    2c-c^2     static and dynamic with the spin-component factors of the
               potential (scaling='hylleraas'; SCS and SOS only)

GTH-PADE pseudopotential, a = 3.5668 Angstrom, Gaussian density fitting.
SCS: c_os = 1.2, c_ss = 1/3.  SOS: c_os = 1.3.

Gaps on a series of meshes have to be extrapolated, e.g. with
a + b Nk^(-1/3) + c Nk^(-2/3); see docs/kpoints.md.  The three-index file
takes 3.9 GB for gth-dzvp 4x4x4 and 14.8 GB for 5x5x5.
'''

import sys
import warnings

from pyscf.pbc import gto, scf
from pycmf.pbc import KOBMP2, kqp

warnings.filterwarnings('ignore')
HARTREE = 27.211386
A0 = 3.5668
# name: (c_os, c_ss)
VARIANTS = {'KOBMP2': (1., 1.), 'SCS-KOBMP2': (1.2, 1. / 3), 'SOS-KOBMP2': (1.3, 0.)}


def edges(se, e_qp):
    '''(direct gap at the first k-point, gap from there to the lowest
    conduction band on the mesh) in eV.'''
    top, bottom = kqp.band_edges(se, e_qp)
    return (bottom[0] - top[0]) * HARTREE, (bottom.min() - top[0]) * HARTREE


def report(label, se, reference, schemes=('static', 'linear', 'dynamic')):
    for scheme in schemes:
        e_qp, z = kqp.quasiparticle(se, scheme, reference)
        print('  %-30s %14.3f %14.3f' % (('%s, %s' % (label, scheme),) + edges(se, e_qp)))
    nocc = int(se[0]['occ'].sum())
    print('  %-30s Z = %.3f (valence), %.3f (conduction) at Gamma' % ('', z[0][nocc - 1], z[0][nocc]))


if __name__ == '__main__':
    basis = sys.argv[1] if len(sys.argv) > 1 else 'gth-szv'
    mesh = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '2x2x2').split('x')]
    exxdiv = None if len(sys.argv) > 3 and sys.argv[3] == 'none' else 'ewald'
    cell = gto.M(atom='C 0 0 0; C %f %f %f' % (A0 / 4, A0 / 4, A0 / 4),
                 a=[[0, A0 / 2, A0 / 2], [A0 / 2, 0, A0 / 2], [A0 / 2, A0 / 2, 0]],
                 basis=basis, pseudo='gth-pade', verbose=0)
    mf = scf.KRHF(cell, cell.make_kpts(mesh), exxdiv=exxdiv).density_fit().run(conv_tol=1e-9)
    print('%s %s exxdiv=%-5s                 direct at Gamma   Gamma -> lowest' % (basis, mesh, exxdiv))

    mp = KOBMP2(mf).init_from_scf()
    se = mp.self_energy(nvir=1, nocc=1)
    print('  %-30s %14.3f %14.3f' % (('HF',) + edges(se, [s['e'] for s in se])))
    report('MP2 on HF', se, 'hf')

    for name, (c_os, c_ss) in VARIANTS.items():
        mp = KOBMP2(mf)
        mp.c_os, mp.c_ss, mp.conv_tol = c_os, c_ss, 1e-7
        mp.single_precision = True
        mp.kernel()
        se = mp.self_energy(nvir=1, nocc=1)
        print('  %-30s %14.3f %14.3f' % ((name,) + edges(se, [s['e'] for s in se])))
        report(name, se, 'obmp2')
        if (c_os, c_ss) != (1., 1.):
            se = mp.self_energy(nvir=1, nocc=1, scaling='hylleraas')
            report(name + ', 2c-c^2', se, 'obmp2', ('static', 'dynamic'))
