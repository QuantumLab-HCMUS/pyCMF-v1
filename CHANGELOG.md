# Changelog

All notable changes to the `pyCMF` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Semantic Versioning.

## [1.0.0] - unreleased

Rewrite of the molecular methods on a common core, prepared for publication.

### Added
* `pycmf.OODH` (experimental): orbital-optimized double hybrids (B2PLYP, PBE0-DH, ...,
  OO-MP2 as the limit, spin-component scaling), variational in the orbitals with the exact
  orbital gradient; relaxed density. Replaces `oo_dh.py`, whose orbital gradient was not the
  derivative of its energy (missing response of the Kohn-Sham potential to the MP2 density and
  the f-gamma commutator, wrong factors in the 2-RDM terms, MP2 on non-canonical orbitals with
  diagonal energies, DIIS set up but not used): its energy rose above the non-optimized double
  hybrid and did not converge.
* `pycmf.properties.transition` (experimental): transition dipole moment and oscillator
  strength between two separately optimized determinants (generalized Slater-Condon rules,
  symmetric orthogonalization, optional sqrt(2) singlet factor). Port of
  `calc_ob_o2b_transition.py` of the OB-O2B-simple scripts.
* `pycmf.mol.noci` (experimental): NOCI in a basis of state-specific OBMP2/O2BMP2
  determinants with the bare, overlap-symmetric and state-specific-Fock Hamiltonians,
  transition moments, <S^2> and channel weights of the roots; `pycmf.mol.excited`:
  MOM-UHF excited starting points and channel selection. Port of `calc_noci_ob_o2b.py`
  without the private `uobmp2_mom_conv` library.
* `pycmf.core`: one self-consistent driver (DIIS, convergence on the energy and on the
  occupied-virtual block of the correlated Fock matrix), occupation schemes
  (Aufbau, MOM, IMOM), spin-block amplitudes.
* `pycmf.mol`: one implementation of the correlation potential for restricted and
  unrestricted references with exact or density-fitted integrals;
  `OBMP2`, `O2BMP2`, `OBDH`, `O2BDH`.
* `pycmf.downfold.OBDF`: one-body downfolding from a converged OBMP2 object, with
  FCIDUMP export and a replaceable active-space solver.
* `pycmf.properties`: HF-like and MP2-like densities, dipole moments, <S^2>,
  isotropic hyperfine couplings.
* Test suite (`pytest`), continuous integration, runnable examples with reference
  output, documentation in `docs/`.
* `pycmf.pbc.KOBMP2`, `KO2BMP2`: k-point OBMP2 rewritten on the common driver
  (experimental). Three-index density-fitted integrals instead of one four-index
  transformation per k-point triple, no Nk^2 o^2 v^2 array, k-point symmetry
  (space group and time reversal), DIIS, frozen bands, k-dependent occupations,
  amplitude scaling, optional MPI over k-points. See `docs/kpoints.md`.
* `KOBMP2.ipea()`: second-order IP/EA corrections for all k-points and band edges
  (IPEA-KOBMP2), from three-index integrals.
* `pycmf.pbc.kqp`, `KOBMP2.self_energy()`, `KOBMP2.quasiparticle_gap()`: diagonal
  second-order self-energy on a frequency grid; static (IPEA), linearized and
  dynamic quasiparticle energies on the OBMP2 or the Hartree-Fock reference;
  `scaling='hylleraas'` for the spin-component factors of the potential.
* `KOBMP2.single_precision` (storage of the three-index tensors) and restart from
  given orbitals.
* `pycmf.pbc.kpeom`, `KOBMP2.peom_mp2()`: partitioned EOM-MP2 for the band edges
  (second-order self-energy with amplitude-dressed integrals) on the orbitals
  of the KOBMP2 object, from three-index integrals, with the symmetry of the
  target k-point; validated against the EOM-KCCSD code of PySCF.
* `KOBMP2.direct_exchange`: Hartree-Fock exchange from the same pass over the
  three-index integrals as the amplitudes (one pass over the integral file less
  per cycle).
* `pycmf.core.mpi`: optional mpi4py helpers.
* `OBDH(mf, xc=...)`: double hybrids by name (B2PLYP, mPW2PLYP, B2T-PLYP, B2GP-PLYP,
  B2K-PLYP, PBE0-DH, PBE-QIDH, PBE0-2), besides the four parameters or a PySCF
  functional string; `set_functional()`, `pycmf.mol.obdh.FUNCTIONALS`.
* `pycmf.pbc.KOBDF`, `KOBMP2.downfold()`: k-point downfolding rewritten on the
  three-index integrals and the correlation potential of KOBMP2 (experimental):
  `get_h1eff`, `get_cderi`, `get_h2eff`; mean field (`scf`) and second-order
  quasiparticle energies (`self_energy`, `quasiparticle`) of the active-space
  Hamiltonian; real supercell Hamiltonian for many-body solvers (`supercell`,
  `as_scf`, `kernel`). Consistent Madelung term with `exxdiv='ewald'`.
* `pycmf.pbc.kqp.tabulate`: the self-energy tabulation shared by KOBMP2 and KOBDF.
* Molecular integral backend accepts mean-field objects with user-defined integrals
  (`mf._eri`).

### Changed
* Package layout by kind of system (`mol`, `pbc`) instead of by method name; algorithm
  variants (slow, einsum, DF, MOM, DIIS, SCS) are attributes, not modules.
* `bch_order=2` is the default for all methods.
* Spin scaling acts on the amplitudes.
* Calculations no longer modify the mean-field object.
* License changed to Apache-2.0.
* RO-OBMP2 and the projection embedding were re-implemented on the new core and
  reproduce the previous energies; the unrestricted k-point OBMP2 was moved to
  `pycmf.pbc` unchanged apart from shared helpers. All are marked experimental.
* k-point OBMP2 no longer overwrites the orbitals of the mean-field object, uses
  PySCF-style attribute names (`max_cycle`, `conv_tol`, `level_shift`, `e_tot`; the
  old names remain as aliases), and `kernel()` returns the energy. The debugging
  output and the ionization/affinity routine with hard-coded band indices are gone.
* Embedding: the subsystem and the environment have separate functionals
  (`xc`/`ax`/`ac` and `xc_env`), as in wave-function-in-DFT embedding; a converged
  Kohn-Sham object of the full system is used as the low level; `make_rdm1` and
  `dip_moment` refer to the full system. `pycmf.embed.partition` holds the SPADE
  partition.

### Removed
* `*_slow`, `*_einsum`, `*_mom`, `*_mom_diis`, `*_scs`, `*_cas` modules and the empty
  `OBCC` package. See `docs/status.md` and `docs/migration.md`.

## [0.1.0] - 2026-03-15 (Major Architectural Refactoring)

This release marks a massive architectural overhaul of the `pyCMF` library. The codebase has been transitioned from a basic two-folder structure into a highly modular, professional package conforming to `pySCF` standards.

### Changed
* **Directory Restructuring:** Completely refactored the original `OBMP2` and `UOBMP2` directories into 5 topic-specific research modules:
  * `OBMP`: Standard Orbital-Optimized MP2 algorithms (incore, restricted, unrestricted, MOM, CAS, Downfolding).
  * `OBDF`: Density Fitting approximations for RAM-reduced OBMP2 calculations.
  * `OBDH`: Orbital-Optimized Double Hybrid DFT methods (e.g., B2PLYP).
  * `KOBMP`: Periodic boundary condition (k-point) OBMP2 calculations.
  * `OBCC`: Orbital-Optimized Coupled Cluster module.
* **File Naming Conventions:** Standardized file names to strictly follow academic and `pySCF` naming conventions. 
  * Removed subjective, ambiguous suffixes (e.g., `_faster`, `_slower`, `_new`, `_mod`, `_ram_reduced`).
  * Adopted clear, descriptive suffixes: `_slow` (pedagogical loops), `_einsum` (vectorized), `_cas` (Complete Active Space), `_mom` (Maximum Overlap Method), `_diis` (SCF convergence), and `_downfold` (Quantum Downfolding).
  * Corrected the misnomer `dfold` to `_downfold` to accurately reflect its quantum downfolding nature rather than density fitting.

### Added
* **Centralized Facade API:** Created a unified `src/pycmf/__init__.py` to handle all user-facing imports. Users can now call state-of-the-art algorithms directly (e.g., `from pycmf import OBMP2, DFUOBMP2`) without needing to know internal file paths.
* **Public APIs (`__all__`):** Implemented strict `__init__.py` files inside each of the 5 modules to define clear public APIs and prevent namespace collisions.
* **Modern Tooling Configurations:** Integrated `ruff` (linter/formatter) and `coverage` configurations directly into `pyproject.toml`. The configurations are specifically tailored to ignore math-heavy whitespace rules and pedagogical `_slow.py` scripts.
* **Project Documentation:** Added standard contribution guidelines (`CONTRIBUTING.md`) outlining the 90/10 Python/C and Functional/OOP philosophies.