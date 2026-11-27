'''Reference implementation for the tests: determinants expanded in the
full CI space of an orthonormal orbital basis.

A determinant with occupied orbitals C (AO) has the CI coefficient
det(M[K, :]) on the string K of the orthonormal basis mo, with M = mo^T S C.
Matrix elements then follow from PySCF's FCI routines without any
Slater-Condon rule, which makes this an independent check of
pycmf.properties.transition and pycmf.mol.noci.
'''

import itertools

import numpy
from pyscf import ao2mo, fci, scf
from pyscf.fci import cistring


def ci_vector(occ_a, occ_b, mo, s1e):
    norb = mo.shape[1]
    out = []
    for occ in (occ_a, occ_b):
        m = mo.T.dot(s1e).dot(occ)
        n = occ.shape[1]
        strings = cistring.make_strings(range(norb), n)
        vec = numpy.empty(len(strings))
        for k, s in enumerate(strings):
            orbs = [p for p in range(norb) if s >> p & 1]
            vec[k] = numpy.linalg.det(m[orbs, :])
        out.append(vec)
    return numpy.outer(out[0], out[1])


class FCISpace:
    '''Operators of a molecule in the FCI space of the orthonormal basis mo.'''

    def __init__(self, mol, mo, nelec):
        self.mol = mol
        self.mo = mo
        self.norb = mo.shape[1]
        self.nelec = nelec
        self.s1e = mol.intor('int1e_ovlp')
        h1 = mo.T.dot(scf.hf.get_hcore(mol)).dot(mo)
        eri = ao2mo.full(mol, mo)
        self.h2e = fci.direct_spin1.absorb_h1e(h1, eri, self.norb, nelec, .5)
        with mol.with_common_orig((0, 0, 0)):
            self.r = numpy.einsum('pi,xpq,qj->xij', mo, mol.intor('int1e_r', comp=3), mo)
        self.dip_nuc = numpy.einsum('a,ax->x', mol.atom_charges(), mol.atom_coords())

    def vector(self, occ):
        return ci_vector(occ[0], occ[1], self.mo, self.s1e)

    def overlap(self, cl, cr):
        return float(numpy.dot(cl.ravel(), cr.ravel()))

    def hamiltonian(self, cl, cr):
        hc = fci.direct_spin1.contract_2e(self.h2e, cr, self.norb, self.nelec)
        return float(numpy.dot(cl.ravel(), hc.ravel())) + self.mol.energy_nuc() * self.overlap(cl, cr)

    def dipole(self, cl, cr):
        dm = fci.direct_spin1.trans_rdm1(cl, cr, self.norb, self.nelec)  # <l|p^+ q|r>
        return self.dip_nuc * self.overlap(cl, cr) - numpy.einsum('xpq,pq->x', self.r, dm)

    def spin_square(self, cl, cr):
        sc = fci.spin_op.contract_ss(cr, self.norb, self.nelec)
        return float(numpy.dot(cl.ravel(), sc.ravel()))

    def matrices(self, occs):
        vecs = [self.vector(o) for o in occs]
        n = len(vecs)
        S, H, S2 = (numpy.empty((n, n)) for _ in range(3))
        D = numpy.empty((3, n, n))
        for i, j in itertools.product(range(n), repeat=2):
            S[i, j] = self.overlap(vecs[i], vecs[j])
            H[i, j] = self.hamiltonian(vecs[i], vecs[j])
            D[:, i, j] = self.dipole(vecs[i], vecs[j])
            S2[i, j] = self.spin_square(vecs[i], vecs[j])
        return S, H, D, S2


def loewdin_pair(cl, cr):
    '''Symmetrically orthogonalized pair of CI vectors.'''
    s = numpy.dot(cl.ravel(), cr.ravel())
    x = 1 / numpy.sqrt(1 + s)
    y = 1 / numpy.sqrt(1 - s)
    c1, c2 = (x + y) / 2, (x - y) / 2
    return c1 * cl + c2 * cr, c2 * cl + c1 * cr
