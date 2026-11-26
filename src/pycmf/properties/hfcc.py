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
Isotropic (Fermi-contact) hyperfine coupling constants.

    A_iso(N) = (4 pi / 3) (g_e g_N mu_B mu_N alpha^2 / <S_z>) rho_s(R_N)

evaluated with the spin density of the OBMP2 determinant or with the
MP2-like spin density.  Values are returned in MHz.
'''

import numpy
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor


def _fermi_contact_integrals(mol, atm_id):
    ao = mol.eval_gto('GTOval', mol.atom_coord(atm_id).reshape(1, 3))
    return 4 * numpy.pi / 3 * numpy.einsum('ip,iq->pq', ao, ao)


def isotropic(mp, atoms=None, correlated=False):
    '''Isotropic hyperfine couplings of the given atoms (default: all).'''
    mol = mp.mol
    if mol.spin == 0:
        raise ValueError('Hyperfine couplings need an open-shell molecule.')
    if atoms is None:
        atoms = range(mol.natm)
    dma, dmb = mp.make_rdm1(ao_repr=True, correlated=correlated)
    spindm = dma - dmb

    effspin = mol.spin * .5
    e_gyro = .5 * nist.G_ELECTRON
    nuc_mag = .5 * (nist.E_MASS / nist.PROTON_MASS)
    au2mhz = nist.HARTREE2J / nist.PLANCK * 1e-6
    fac = nist.ALPHA**2 / 2 / effspin * e_gyro * au2mhz

    out = []
    for ia in atoms:
        nuc_gyro = get_nuc_g_factor(mol.atom_symbol(ia)) * nuc_mag
        rho = numpy.einsum('ij,ji', _fermi_contact_integrals(mol, ia), spindm)
        out.append(2 * fac * nuc_gyro * rho)
    return numpy.array(out)
