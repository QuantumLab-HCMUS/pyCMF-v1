# Status of the methods and known issues

Version 1.0.0 is a rewrite of the molecular part of pyCMF 0.2 on a common
core. This page states what has been checked, what has not, and what changed
with respect to 0.2.

## Maturity

| Method | Module | Checked against | Open points |
| --- | --- | --- | --- |
| OBMP2 (R, U; exact and DF integrals) | `mol/obmp2.py` | MP2 at the first cycle; 0.2 energies to 1e-10 Ha | no frozen core, no analytic gradients |
| O2BMP2 | `mol/obmp2.py` | 0.2 energies (amplitude scaling) | no O(N⁴) Laplace algorithm; no recommended `c_os` |
| state-specific (MOM, IMOM) | `core/occupation.py` | 0.2 energies | open-shell singlets are spin-mixed determinants |
| OBDF | `downfold/obdf.py` | FCI and OBMP2 limits; 0.2 energies | FCIDUMP for restricted references only |
| OBDH, O2BDH | `mol/obdh.py` | OBMP2 and hybrid-DFT limits; 0.2 energies | parameters of the published double hybrids, not re-optimised; no published benchmark |
| OO-DH (orbital-optimized double hybrid) | `mol/oodh.py` | orbital gradient = finite differences (1e-8); ORCA 4.1 OO-RI-MP2 (6e-8 Eh) and OO-RI-B2PLYP for H₂O and NH₂ (orbital-optimization lowering to 1e-8 Eh); hybrid-DFT limit; relaxed dipole = finite field | **experimental**: unrestricted only, four-index blocks in memory (no batching), no analytic nuclear gradients, no frozen core |
| RO-OBMP2 | `mol/roobmp2.py` | 0.2 energies to 1e-10 Ha | **experimental**: no formal derivation, no DIIS, no benchmark |
| embedding + concentric localization | `embed/` | 0.2 energies to 1e-10 Ha; whole system = high-level method; DFT-in-DFT = Kohn–Sham | **experimental**: unrestricted formulation only; no published benchmark |
| k-point OBMP2 and O2BMP2 (closed shell) | `pbc/kobmp2.py`, `kbch.py`, `kintegrals.py`, `ksymm.py` | previous four-index implementation, cycle by cycle, to 1e-10 Ha; KMP2 at the first cycle; molecular code at the Γ point | **experimental**: rewritten (see [kpoints.md](kpoints.md)), no published benchmark yet |
| k-point OBMP2 (unrestricted) | `pbc/kuobmp2.py` | — | **experimental**: known inconsistency, see below |
| transition dipole between two state-specific determinants | `properties/transition.py`, `mol/excited.py` | full-CI expansion of the two determinants (Löwdin pair, 1e-9); unrelaxed single excitation = −⟨i\|r\|a⟩; OB-O2B-simple scripts with the 0.2 library (H₂O/cc-pVDZ, 1e-7) | **experimental**: determinant-level moment, no correlated transition density; `singlet=True` is the √2 approximation |
| NOCI of state-specific OBMP2/O2BMP2 determinants | `mol/noci.py` | S, H, dipole, S² against full CI (1e-9), also with a singular orbital overlap; OB-O2B-simple scripts with the 0.2 library (matrices 1e-13, bare NOCI energies 1e-7) | **experimental**: the effective Hamiltonians are model choices, see known issue 8 |
| k-point OBDF | `pbc/kobdf.py` | four-index KROBDF of the group (branch `thai`, 67a4494) to 1e-8 Ha; molecular OBDF at the Γ point; complete and empty active space; sum rule | **experimental**: rewritten on the integrals of KOBMP2 (see [kpoints.md](kpoints.md)); closed shell; the Wannier90, band and DOS tools of the branch are not ported |

"0.2 energies" means total energies of the previous implementation
(master branch, commit 29efd04) on small molecules; they are regression
values, not an independent validation. None of the published benchmark
numbers has been re-run with this version yet.

## Known issues

1. **KUOBMP2 does not reproduce KOBMP2 for a closed-shell cell.** For
   diamond (gth-szv, two k-points) KOBMP2 gives −10.6408 Ha and KUOBMP2
   −8.6381 Ha from the same Hartree–Fock energy (−10.5311 Ha). The moved
   code reproduces the 0.2 numbers exactly, so the discrepancy is in the
   method code, not in the restructuring.
2. **Hard-coded band indices in `kuobmp2.py`.** The ionization
   and affinity analysis addresses orbitals `nocc-3 … nocc+1`; cells with
   fewer than three occupied or two virtual bands per k-point raise
   `IndexError`. `KOBMP2` no longer has this code: it reports the band edges
   of the correlated Fock matrix (`ip_koopmans`, `ea_koopmans`, `band_gap`).
3. **KUOBMP2 is still the four-index code of 0.2.** It has the cost
   problems that `KOBMP2` had (see [kpoints.md](kpoints.md)) and should be
   rebuilt on `pbc/kbch.py`.
4. **Embedding functional: the two levels are separate.** In 0.2 the
   functional of the environment (`xc_env`) was also used as the Kohn–Sham
   part of the embedded "OBDH", so `xc_env='pbe'` meant PBE plus `a_c` times
   the OBMP2 correlation, with full PBE correlation. In 1.0 the convention of
   wave-function-in-DFT embedding is used throughout: the environment and the
   embedding potential use `xc_env`, and the subsystem uses its own double
   hybrid (`xc`, or `ax`, `ac`, `xc_x`, `xc_c`; B2PLYP by default). The two
   conventions give very different energies (−152.5729 against −152.6107 Ha
   for the water dimer of the tests). The 0.2 numbers are reproduced with
   `xc=xc_env` and are kept as a regression test only.
5. **KROBDF with `exxdiv='ewald'`.** With `exxdiv=None` the rewritten
   `KOBDF` reproduces the constant, the one-body integrals and the mean-field
   bands of KROBDF. With `exxdiv='ewald'` the one-body integrals agree, but
   the constant of KROBDF is higher by $\xi N_{\rm act}$ ($\xi$ the Madelung
   constant of the k-mesh, $N_{\rm act}$ the active electrons per cell) than
   the value for which the reference determinant has the energy
   $E_{\rm KOBMP2}-E_c[T_{\rm act}]$, and its active Fock matrix has no
   shift of the occupied levels. `KOBDF` puts $-\xi N_{\rm act}/2$ in the
   constant and the shift in `get_fock`; both limits (complete and empty
   active space) then hold for either setting.
6. **Degenerate open shells with OBDH.** For states such as the ²Π state of
   OH the orientation of the open-shell orbital is fixed only by numerical
   noise of the DFT grid. Convergence then stalls at $|\bar f_{ia}|\sim10^{-6}$
   and total energies are reproducible to about 1e-7 Ha only.
7. **The energy is not variational**, so the error in the energy is first
   order in the residual $\bar f_{ia}$. Use a tight `conv_tol_grad` when
   energy differences below 1e-6 Ha matter.
8. **NOCI with `hamiltonian='state_specific_fock'` does not couple the α and β
   branches of a channel.** These two determinants differ by two spin
   orbitals, and the averaged Fock operator is a one-body operator. Their
   coupling is therefore only what the orbital non-orthogonality allows: the
   pairs stay (nearly) degenerate, singlet and triplet are not separated, and
   μ, f and ⟨S²⟩ of a degenerate pair depend on an arbitrary rotation.
   Example: H₂O/cc-pVDZ, 4→5, gives two roots at 8.0759 eV with ⟨S²⟩ ≈ 1.
   The `bare` and `overlap_symmetric` models keep the two-electron coupling
   (triplet 6.72/7.73 eV, singlet 7.42/8.42 eV). The original scripts show
   the same degeneracy.
9. **The OB-O2B-simple scripts with the 0.2 library `uobmp2_mom_conv`.**
   These problems were found while porting the scripts; they are absent from
   `pycmf.properties.transition` and `pycmf.mol.noci`:
   - The 2025 copies of the library (`pyscf-ob-mp2-dft`, `Downloads/pyscf-ob-mp2`)
     never read `css`/`cos`, so "O2BMP2" there is OBMP2.
   - After convergence `mo_coeff` is replaced by one more diagonalization
     (energy order, no MOM), so `mo_occ > 0.5` can select a different
     determinant from the optimized one. For H₂O/cc-pVDZ, channel 3→6, the
     stored determinant has zero overlap with the optimized one while the
     energy is right. The Fock rebuild then fails (0.40 Eh) and the bare NOCI
     gets two spurious roots near 30 eV.
   - Convergence on the energy alone leaves |f_ov| ≈ 1e-5. This changes the
     NOCI couplings by about 1e-4 Eh and breaks symmetry-forbidden couplings.

## Changes with respect to 0.2

- One class per method; variants are attributes. The modules `*_slow`,
  `*_einsum`, `*_mom`, `*_mom_diis`, `*_scs`, `*_cas`, `df*` are gone.
- `bch_order=2` is the default everywhere. In 0.2 `OBMP2` defaulted to the
  first-order variant and `DFOBMP2` to second order.
- Spin scaling acts on the amplitudes, as in the O2BMP2 paper. The different
  scaling of `uobmp2_scs.py` was not carried over.
- The mean-field object is no longer modified by a calculation.
- Convergence is tested on the energy **and** on $\bar f_{ia}$.
- DIIS uses the commutator $[F,D]$ per spin; the history is limited to
  `diis_space` vectors.
- The integral transformation is restricted to the blocks that are needed.
- `frozen` orbitals raise `NotImplementedError` instead of failing later.
- Not carried over: the second-order correction to Koopmans ionization
  potentials and electron affinities (`make_IPEA`), the CAS variants
  (`obmp2_cas.py`, `uobmp2_cas.py`, `uobmp2_cas_scf.py`), the unused
  `dft_uobmp2.py` / `OBDH_CL_in_DFT.py` of the `kdfold` branch, and the
  `stabilize_scf` helper (PySCF's `mf.stability()` covers it).
- Not ported from the branch `thai`: the Wannier90 interface, the real-space
  Hamiltonian H(R), band-structure and DOS plotting of the KROBDF package.
  `KOBDF` provides what they start from: `get_h1eff()`, `get_fock()`, the
  mean-field and quasiparticle bands on the k-mesh, and the active orbitals.
- The 0.2 example inputs are kept in `benchmarks/legacy-0.2/`; they use the
  old interface and have to be ported before use.
