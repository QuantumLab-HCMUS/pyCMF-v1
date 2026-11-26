# Usage

All stable methods follow PySCF conventions: build the object from a converged
mean-field object, set attributes, call `kernel()` (or `run()`), read results
from attributes.

## Choosing the method

| Call | Reference | Result |
| --- | --- | --- |
| `pycmf.OBMP2(mf)` | RHF/RKS → `ROBMP2`; UHF/ROHF → `UOBMP2` | OBMP2 |
| `pycmf.O2BMP2(mf, c_os=1.2)` | same | O2BMP2; `c_os` is mandatory |
| `pycmf.OBDH(mf, xc='B2PLYP')` | same → `ROBDH` / `UOBDH` | one-body double hybrid |
| `pycmf.O2BDH(mf, c_os=1.2)` | same | spin-opposite-scaled OBDH |
| `pycmf.OBDF(mp, ncas, nelecas)` | converged OBMP2/O2BMP2 | active-space Hamiltonian |

Keyword arguments of the factory functions are set as attributes, so
`pycmf.OBMP2(mf, conv_tol=1e-10, level_shift=0.1)` is the same as setting them
afterwards. A ROHF reference is converted to UHF form.

## Attributes

| Attribute | Default | Meaning |
| --- | --- | --- |
| `conv_tol` | `1e-8` | energy convergence |
| `conv_tol_grad` | `sqrt(conv_tol)` | threshold on the norm of $\bar f_{ia}$ |
| `max_cycle` | `100` | `max_cycle=1` from HF orbitals gives conventional MP2 |
| `bch_order` | `2` | `1` for the half-weighted first commutator |
| `c_os`, `c_ss` | `1, 1` | scaling of opposite-spin / same-spin amplitudes |
| `level_shift` | `0` | added to the energy gap in the denominators |
| `occupation` | `'aufbau'` | `'mom'` or `'imom'` for state-specific calculations |
| `diis`, `diis_space`, `diis_start_cycle` | `True, 8, 1` | DIIS on the correlated Fock matrix |
| `with_df` | from `mf` | density-fitting object; `None` = exact integrals |
| OBDH: `xc`, `ax`, `ac`, `xc_x`, `xc_c`, `grids_level` | B2PLYP | functional definition, see below |

Results: `converged`, `e_tot`, `e_corr` (relative to the input mean field),
`e_ref` (HF or KS energy of the final determinant), `mo_energy`, `mo_coeff`,
`mo_occ`. Orbitals are ordered occupied first. The input `mf` is never
modified.

## Double-hybrid functionals

```python
pycmf.OBDH(mf)                                          # B2PLYP
pycmf.OBDH(mf, xc='B2GP-PLYP')                          # by name
pycmf.OBDH(mf, ax=0.6, ac=0.3, xc_x='PBE', xc_c='PBE')  # by parameters
pycmf.OBDH(mf, xc='0.5*HF + 0.5*B88, 0.75*LYP', ac=0.25)   # PySCF string + a_c
pycmf.O2BDH(mf, c_os=1.2, xc='PBE0-DH')
```

Names (case, hyphens and underscores are ignored): B2PLYP, mPW2PLYP,
B2T-PLYP, B2GP-PLYP, B2K-PLYP, PBE0-DH, PBE-QIDH, PBE0-2
(`pycmf.mol.obdh.FUNCTIONALS`). The semilocal parts are evaluated by PySCF
(libxc). `m.get_xc()` returns the functional string of the Kohn–Sham part,
`m.ax` and `m.ac` the two weights.

## Density fitting

- `mf = scf.RHF(mol).density_fit()` → pyCMF reuses `mf.with_df` (JK-fit basis)
  for the correlation part, as PySCF's DF-MP2 does.
- `pycmf.OBMP2(mf).density_fit()` on an exact-integral reference fits only the
  correlation part with an RI-C basis; `density_fit(auxbasis='cc-pvtz-ri')`
  selects one.

Exact integrals need the AO integrals in memory. For more than a few hundred
basis functions use density fitting.

## Excited states

```python
mf_ex = scf.addons.mom_occ(scf.UHF(mol), mf_gs.mo_coeff, occ)   # non-Aufbau HF
mf_ex.kernel(mf_ex.make_rdm1(mf_gs.mo_coeff, occ))
ex = pycmf.O2BMP2(mf_ex, c_os=1.2, occupation='imom').run()
```

`mo_occ` of the mean-field object defines the starting determinant; orbitals
are reordered occupied-first internally. At `verbose >= 5` the number of
electrons that left the reference occupied space is printed every cycle, which
shows a collapse to another state.

## Downfolding

```python
ham = pycmf.OBDF(mp, ncas=6, nelecas=6)              # orbitals around the Fermi level
ham = pycmf.OBDF(mp, 4, (1, 1), cas_list=[0, 1, 4, 5])   # 0-based indices in mp.mo_coeff
e = ham.kernel()                 # FCI; ham.fcisolver can be replaced, ham.nroots for excited states
h1, e_core = ham.get_h1eff()     # dressed one-body integrals and constant
h2 = ham.get_h2eff()             # bare active-space integrals (chemists' notation)
ham.to_fcidump('active.fcidump') # restricted reference only
```

For an unrestricted reference `h1` is a pair and `h2` a triple
(αα, αβ, ββ). Do not cut through a set of degenerate orbitals with
`cas_list`: the result then depends on an arbitrary rotation.

## Properties

```python
mp.make_rdm1(ao_repr=True, correlated=False)   # HF-like; correlated=True: MP2-like
mp.dip_moment(correlated=True)
mp.spin_square(correlated=False)               # unrestricted only
mp.hfcc(atoms=None, correlated=False)          # isotropic hyperfine couplings, MHz
mp.ip_koopmans(); mp.ea_koopmans()
mp.get_fock(); mp.get_vcorr()                  # correlated Fock matrix, correlation potential (MO)
```

## Experimental methods

```python
from pycmf.mol.roobmp2 import ROOBMP2          # ROHF reference
from pycmf.embed import EmbeddedOBDH, EmbeddedOBMP2
from pycmf.pbc import KOBMP2, KO2BMP2          # k-point sampling, closed shell
from pycmf.pbc import KOBDF                    # k-point downfolding
from pycmf.pbc import KUOBMP2                  # code of version 0.2
```

See [status.md](status.md) before using them.

### Embedding

```python
from pycmf.embed import EmbeddedOBDH, EmbeddedOBMP2

ks = dft.UKS(mol, xc='pbe').density_fit().run()        # low level, full system
emb = EmbeddedOBDH(ks, active_atoms=[0, 1, 2], xc='B2PLYP', use_cl=True, n_shells=1)
e = emb.kernel()
emb.e_env, emb.e_sub, emb.nelec_sub            # E(PBE, full), E(high level, A), electrons of A
emb.make_rdm1(); emb.dip_moment()              # full system (with_env=False: subsystem)
EmbeddedOBMP2(ks, active_atoms=[0, 1, 2]).kernel()
```

The environment and the embedding potential use the functional of the
Kohn–Sham object (or `xc_env`; with a Hartree–Fock object a PBE calculation
is run). The subsystem uses its own functional, chosen as for `OBDH`.

### Periodic systems

```python
from pyscf.pbc import gto, scf
from pycmf.pbc import KOBMP2, KO2BMP2

mf = scf.KRHF(cell, cell.make_kpts([4, 4, 4])).density_fit().run()
mp = KOBMP2(mf).run()                          # energies per unit cell
mp.e_tot, mp.band_gap(), mp.mo_energy          # bands of the correlated Fock matrix
KO2BMP2(mf, c_os=1.2).run()
KOBMP2(mf, frozen=1, kpt_symmetry=False)       # frozen lowest band, full k-mesh
```

The attributes are those of the molecular classes (`conv_tol`, `max_cycle`,
`diis`, `bch_order`, `c_os`, `c_ss`, `level_shift`). The mean field must be
a closed-shell `KRHF`; with FFT-based integrals in the mean field a Gaussian
density fitting object is built for the correlation part. Algorithm, MPI
usage and timings are described in [kpoints.md](kpoints.md).

### Downfolding with k-points

```python
mp = KOBMP2(mf).run()
ham = mp.downfold(ncas=8, nelecas=8)           # KOBDF(mp, 8, 8); cas_list=[...] for other bands
h1, e_core = ham.get_h1eff()                   # (nkpts, ncas, ncas), constant per cell
ham.get_cderi(k1, k2)                          # three-index integrals of the active orbitals
ham.get_h2eff(k1, k2, k3)                      # (t k1, u k2 | v k3, w k4) / Nk
ham.scf(); ham.mo_energy; ham.band_gap()       # mean field of the model
ham.self_energy(nocc=1, nvir=1)
ham.quasiparticle_gap('dynamic')               # second-order gap of the model
h1s, h2s, e0, nelec = ham.supercell()          # real Hamiltonian of the supercell
ham.as_scf()                                   # PySCF RHF object for CCSD, CASCI, DMRG, ...
ham.kernel()                                   # FCI of the supercell, energy per cell
```

`KOBMP2(mf).init_from_scf().downfold(...)` gives the second-order downfolding
on Hartree–Fock orbitals. See [kpoints.md](kpoints.md).
