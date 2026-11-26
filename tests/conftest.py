'''Shared molecules and reference data for the pyCMF test suite.

Unless stated otherwise, reference energies were produced with the original
pyCMF implementation (commit 29efd04 of the master branch) and are used here
as regression values for the rewritten code.
'''

import pytest
from pyscf import gto, scf

GEOM = {
    'h2o': ('O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692', 0),
    'oh': ('O 0 0 0; H 0 0 0.97', 1),
    'ch2': ('C 0 0 0; H 0 0.99 0.42; H 0 -0.99 0.42', 2),
    'nh2': ('N 0 0 0; H 0 0.80 0.60; H 0 -0.80 0.60', 1),
    'lih': ('Li 0 0 0; H 0 0 1.6', 0),
}


def make_mol(name, basis='6-31g'):
    atom, spin = GEOM[name]
    return gto.M(atom=atom, basis=basis, spin=spin, verbose=0)


@pytest.fixture(scope='session')
def mf_cache():
    '''Converged mean-field objects, computed once per session.'''
    cache = {}

    def get(name, kind='uhf', df=False, basis='6-31g'):
        key = (name, kind, df, basis)
        if key not in cache:
            mol = make_mol(name, basis)
            mf = {'rhf': scf.RHF, 'uhf': scf.UHF, 'rohf': scf.ROHF}[kind](mol)
            if df:
                mf = mf.density_fit()
            mf.conv_tol = 1e-12
            cache[key] = mf.run()
        return cache[key]
    return get
