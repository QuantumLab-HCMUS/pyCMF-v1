#!/usr/bin/env python
'''
Accuracy of Laplace-transformed amplitudes in self-consistent O2BMP2.

The energy denominators of the amplitudes are replaced by the quadrature

    1 / (e_i + e_j - e_a - e_b) = - sum_m w_m exp[(e_i - e_a) t_m] exp[(e_j - e_b) t_m],

which is what a Laplace-transformed, density-fitted O(N^4) algorithm for the
opposite-spin terms would use; everything else is the production code.  The
points and weights are a least-squares fit of the relative error on the range
of denominators of the molecule.  Printed: error of the total energy and of
the orbital energies against the exact denominators, for 4 to 12 points.

Result (H2O/cc-pVTZ, C2H4/cc-pVDZ, c_os = 1.2): below 1e-7 Ha with 8 points.
The quadrature needs denominators of one sign, i.e. Aufbau occupations.
'''
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import least_squares
from pyscf import gto, scf
import pycmf
from pycmf.core import amplitudes
H = 27.211386

def laplace_quadrature(xmin, xmax, n):
    '''1/x ~ sum_m w_m exp(-x t_m) on [xmin, xmax]: least-squares fit of the relative error'''
    x = np.exp(np.linspace(np.log(xmin), np.log(xmax), 300))
    t0 = np.exp(np.linspace(np.log(.3 / xmax), np.log(6. / xmin), n))
    def resid(p):
        t, w = np.exp(p[:n]), np.exp(p[n:])
        return (np.exp(-np.outer(x, t)) @ w) * x - 1
    w0 = np.linalg.lstsq(np.exp(-np.outer(x, t0)) * x[:, None], np.ones_like(x), rcond=None)[0]
    p0 = np.concatenate([np.log(t0), np.log(np.abs(w0) + 1e-8)])
    sol = least_squares(resid, p0, method='lm', max_nfev=4000)
    return np.exp(sol.x[:n]), np.exp(sol.x[n:]), abs(resid(sol.x)).max()

exact_den = amplitudes._denominator
state = {}
def laplace_den(e_ia, e_jb, shift):
    x1, x2 = -e_ia, -e_jb                       # positive
    lo, hi = x1.min() + x2.min(), x1.max() + x2.max()
    t, w, err = laplace_quadrature(lo, hi, state['n'])
    state['err'] = err; state['range'] = hi / lo
    inv = np.einsum('m,iam,jbm->iajb', w, np.exp(-x1[:, :, None] * t), np.exp(-x2[:, :, None] * t))
    return -1. / inv

for name, atom, basis in (('H2O', 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692', 'cc-pvtz'),
                          ('C2H4', 'C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321', 'cc-pvdz')):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-11)
    amplitudes._denominator = exact_den
    ref = pycmf.O2BMP2(mf, c_os=1.2); ref.conv_tol = 1e-10; ref.kernel()
    n = mol.nelectron // 2; e0 = np.asarray(ref.mo_energy).ravel()
    print('%s/%s: exact DF O2BMP2 E_corr = %.8f Ha, HOMO %.4f eV' % (name, basis, ref.e_tot - mf.e_tot, e0[n - 1] * H))
    for ng in (4, 6, 8, 10, 12):
        state['n'] = ng; amplitudes._denominator = laplace_den
        mp = pycmf.O2BMP2(mf, c_os=1.2); mp.conv_tol = 1e-10; mp.kernel()
        e = np.asarray(mp.mo_energy).ravel()
        print('   %2d points: max rel. error of 1/x %.1e (range %4.0f) | dE %9.2e Ha | max d(eps) %8.1e eV | converged %s' % (
            ng, state['err'], state['range'], mp.e_tot - ref.e_tot, abs(e - e0).max() * H, mp.converged))
amplitudes._denominator = exact_den
