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
One-body double-hybrid methods (OBDH and O2BDH) for molecules.

A double-hybrid functional mixes Hartree-Fock exchange and a second-order
correlation energy into a semilocal functional,

    E = E_KS[a_x HF + (1 - a_x) X, (1 - a_c) C] + a_c E_c^(2).

In OBDH the second-order part is the OBMP2 one and it acts on the orbitals:
the Kohn-Sham Fock matrix is augmented by the scaled OBMP2 correlation
potential,

    fbar = f_KS + a_c v_OBMP2,       E = E_KS + a_c E_c^OBMP2,

and fbar is diagonalised self-consistently.  The amplitudes use the
eigenvalues of fbar; the correlation potential is built with the HF Fock
matrix of the current orbitals.  O2BDH keeps only the opposite-spin
amplitudes, scaled by c_os.

Functional.  The Kohn-Sham part is evaluated by PySCF (libxc) and is defined
in one of three ways:

    OBDH(mf, xc='B2GP-PLYP')                   a double hybrid of FUNCTIONALS
    OBDH(mf, ax=0.6, ac=0.3, xc_x='PBE', xc_c='PBE')     the four parameters
    OBDH(mf, xc='0.5*HF + 0.5*B88, 0.75*LYP', ac=0.25)   any PySCF functional
                                                         string and a_c

The default is B2PLYP.  The parameters of the published double hybrids were
fitted with a perturbative second-order energy on Kohn-Sham orbitals; they
have not been re-optimised for the self-consistent scheme.
'''

from pyscf import dft, scf
from pyscf.lib import logger

from pycmf.mol.obmp2 import ROBMP2, UOBMP2

#: Global double hybrids: name -> (a_x, a_c, semilocal exchange, semilocal correlation).
#: The semilocal correlation enters with the weight 1 - a_c in all of them.
FUNCTIONALS = {
    'B2PLYP': (0.53, 0.27, 'B88', 'LYP'),        # Grimme, J. Chem. Phys. 124, 034108 (2006)
    'MPW2PLYP': (0.55, 0.25, 'mPW91', 'LYP'),    # Schwabe, Grimme, PCCP 8, 4398 (2006)
    'B2TPLYP': (0.60, 0.31, 'B88', 'LYP'),       # Tarnopolsky et al., JPCA 112, 3 (2008)
    'B2GPPLYP': (0.65, 0.36, 'B88', 'LYP'),      # Karton et al., JPCA 112, 12868 (2008)
    'B2KPLYP': (0.72, 0.42, 'B88', 'LYP'),       # Tarnopolsky et al., JPCA 112, 3 (2008)
    'PBE0DH': (0.5, 0.125, 'PBE', 'PBE'),        # Bremond, Adamo, JCP 135, 024106 (2011)
    'PBEQIDH': (3**(-1. / 3), 1. / 3, 'PBE', 'PBE'),   # Bremond et al., JCP 141, 031101 (2014)
    'PBE02': (2**(-1. / 3), 0.5, 'PBE', 'PBE'),  # Chai, Mao, CPL 538, 121 (2012)
}


def _key(name):
    return ''.join(ch for ch in str(name).upper() if ch.isalnum())


def double_hybrid(name):
    '''(a_x, a_c, xc_x, xc_c) of a named double hybrid, or None if ``name``
    is not one of :data:`FUNCTIONALS` (hyphens, underscores and case are
    ignored: 'B2GP-PLYP', 'b2gpplyp', 'PBE0-2', 'PBE-QIDH').'''
    return FUNCTIONALS.get(_key(name))


class _OBDHMixin:
    '''Attributes:
        ax : fraction of HF exchange
        ac : weight of the OBMP2 correlation
        xc_x, xc_c : semilocal exchange and correlation functionals
        xc : name of a double hybrid of :data:`FUNCTIONALS` (it then defines
            ax, ac, xc_x and xc_c), or a functional string of PySCF for the
            Kohn-Sham part (the semilocal correlation must already be scaled
            by 1 - ac; ``ac`` is taken from the attribute)
        grids_level : DFT integration grid level (default: that of PySCF)
    '''
    ax = 0.53
    ac = 0.27
    xc_x = 'B88'
    xc_c = 'LYP'
    xc = None
    grids_level = None

    _obdh_keys = {'ax', 'ac', 'xc_x', 'xc_c', 'xc', 'grids_level'}

    def set_functional(self, name):
        '''Take the four parameters from a named double hybrid.'''
        par = double_hybrid(name)
        if par is None:
            raise KeyError('Unknown double hybrid %r.  Available: %s'
                           % (name, ', '.join(sorted(FUNCTIONALS))))
        self.ax, self.ac, self.xc_x, self.xc_c = par
        self.xc = None
        return self

    def _functional(self):
        '''(functional string of the Kohn-Sham part, a_c)'''
        if self.xc is not None:
            par = double_hybrid(self.xc)
            if par is None:
                return self.xc, self.ac
            ax, ac, xc_x, xc_c = par
        else:
            ax, ac, xc_x, xc_c = self.ax, self.ac, self.xc_x, self.xc_c
        if not (0 <= ax <= 1 and 0 <= ac <= 1):
            raise ValueError('ax and ac must lie in [0, 1]; got %s, %s' % (ax, ac))
        return '%.10g*HF + %.10g*%s, %.10g*%s' % (ax, 1 - ax, xc_x, 1 - ac, xc_c), ac

    def get_xc(self):
        '''Functional string of the Kohn-Sham part.'''
        return self._functional()[0]

    def get_ks(self):
        '''Kohn-Sham object that evaluates the hybrid part.'''
        xc = self.get_xc()
        ks = getattr(self, '_ks', None)
        if ks is None or ks.xc != xc:
            hf = self.get_hf()
            ks = dft.UKS(self.mol) if isinstance(hf, scf.uhf.UHF) else dft.RKS(self.mol)
            ks.xc = xc
            ks.verbose = self.verbose
            ks.stdout = self.stdout
            ks.max_memory = self.max_memory
            if self.grids_level is not None:
                ks.grids.level = self.grids_level
            with_df = getattr(hf, 'with_df', None)
            if with_df:
                ks = ks.density_fit()
                ks.with_df = with_df
            self._ks = ks
        return ks

    def get_reference(self, dm, h1e, vhf, fock_hf, e_hf):
        ks = self.get_ks()
        vxc = ks.get_veff(self.mol, dm)
        e_ks = ks.energy_elec(dm, h1e, vxc)[0] + ks.energy_nuc()
        return self._unpack(h1e + vxc), e_ks, self._functional()[1]

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        xc, ac = self._functional()
        logger.info(self, 'XC functional = %s', xc)
        logger.info(self, 'weight of the OBMP2 correlation a_c = %g', ac)
        return self


class ROBDH(_OBDHMixin, ROBMP2):
    '''OBDH with a closed-shell restricted reference.'''
    _keys = ROBMP2._keys | _OBDHMixin._obdh_keys


class UOBDH(_OBDHMixin, UOBMP2):
    '''OBDH with an unrestricted reference.'''
    _keys = UOBMP2._keys | _OBDHMixin._obdh_keys


def OBDH(mf, ax=None, ac=None, frozen=None, mo_coeff=None, mo_occ=None, xc=None, **kwargs):
    '''One-body double hybrid.  ``mf`` is a converged HF or KS object that
    provides the starting orbitals.

    ``xc`` names a double hybrid (default 'B2PLYP') or is a PySCF functional
    string of the Kohn-Sham part; ``ax`` and ``ac`` given explicitly override
    the parameters of the named functional.  Further keyword arguments are
    set as attributes.
    '''
    if isinstance(mf, scf.uhf.UHF) or isinstance(mf, scf.rohf.ROHF):
        mp = UOBDH(mf, frozen, mo_coeff, mo_occ)
    else:
        mp = ROBDH(mf, frozen, mo_coeff, mo_occ)
    if xc is not None:
        if double_hybrid(xc) is not None:
            mp.set_functional(xc)
        else:
            if ax is not None:
                raise ValueError('ax has no meaning together with a functional string; '
                                 'put the fraction of HF exchange in the string.')
            mp.xc = xc
    if ax is not None:
        mp.ax = ax
    if ac is not None:
        mp.ac = ac
    return mp.set(**kwargs)


def O2BDH(mf, c_os=None, ax=None, ac=None, frozen=None, mo_coeff=None, mo_occ=None,
          xc=None, **kwargs):
    '''Spin-opposite-scaled one-body double hybrid.  ``c_os`` must be given.'''
    if c_os is None:
        raise ValueError('O2BDH needs an explicit c_os (e.g. c_os=1.2).')
    return OBDH(mf, ax, ac, frozen, mo_coeff, mo_occ, xc, c_os=c_os, c_ss=0., **kwargs)
