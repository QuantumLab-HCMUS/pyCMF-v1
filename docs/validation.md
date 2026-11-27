# Validation and performance

## What the test suite checks

`pytest -m "not slow"` runs 137 tests in a few minutes; `pytest` adds nine
slower k-point tests (symmetry, larger meshes, MPI; about two minutes). Tolerances are 5e-9 Ha for regression values unless noted.

| Property | Test | Reference |
| --- | --- | --- |
| First cycle from HF orbitals is MP2 (R, U) and DF-MP2 | `test_obmp2.py` | `pyscf.mp` |
| First-cycle correlated density is the unrelaxed MP2 density | `test_properties.py` | `pyscf.mp` |
| Restricted = unrestricted for closed shells (OBMP2, O2BMP2, OBDH, OBDF) | all files | internal |
| Converged orbital energies are the eigenvalues of the correlated Fock matrix; $\bar f_{ia}=0$ | `test_obmp2.py` | internal |
| DIIS on/off give the same energy | `test_obmp2.py` | internal |
| The mean-field object is not modified | `test_obmp2.py` | internal |
| `c_os = c_ss = 0` is Hartree–Fock; `c_os = c_ss = 1` is OBMP2 | `test_o2bmp2.py` | internal |
| OBDH with $a_x=a_c=1$ is OBMP2; with $a_c=0$ it is hybrid DFT | `test_obdh.py` | `pyscf.dft` |
| OBDF with all orbitals active is FCI; with none it is OBMP2 | `test_obdf.py` | `pyscf.fci` |
| FCIDUMP round trip | `test_obdf.py` | `pyscf.tools.fcidump` |
| Total energies of version 0.2: OBMP2 (BCH 1 and 2, level shift, DF), O2BMP2, SCS-type scaling, MOM excited state, OBDF (5 active spaces, R and U), OBDH/O2BDH | all files | version 0.2 |
| ⟨S²⟩ (determinant and corrected) and hyperfine couplings of OH | `test_properties.py` | version 0.2 |
| RO-OBMP2, embedding with and without concentric localization | `test_experimental.py` | version 0.2 |
| Embedding: all atoms in the subsystem give the high-level method (OBMP2, OBDH); DFT-in-DFT with one functional gives the Kohn–Sham energy | `test_experimental.py` | internal, `pyscf.dft` |
| OO-DH: analytic orbital gradient = finite differences (B2PLYP, PBE0-DH open shell, OO-MP2); OO-MP2 and OO-B2PLYP against ORCA 4.1 (H₂O, NH₂); $a_c=0$ is hybrid DFT; relaxed dipole = finite field | `test_oodh.py` | ORCA 4.1.0, internal |
| Named double hybrids are their four parameters; `'PBE0-2'` is not parsed as a difference | `test_obdh.py` | internal |
| Transition dipole: Löwdin pair in the full CI space; HF limit −⟨i\|r\|a⟩; gs↔es symmetry; restricted = unrestricted ground state; OB-O2B-simple value (H₂O, β 4→5) | `test_transition.py` | full CI, OB-O2B-simple scripts with the 0.2 library |
| NOCI: S, H, dipole, S² of OBMP2, MOM-UHF and unrelaxed determinants against full CI (exact and direct J/K integrals); bare diagonal = HF energy of the OBMP2 determinant; effective diagonals = OBMP2 energies; constant of the effective operator; linear dependencies; bare NOCI energies of the OB-O2B-simple scripts | `test_noci.py` | full CI, internal, OB-O2B-simple scripts with the 0.2 library |
| k-point OBDF: constant, one-body integrals and bands of the four-index KROBDF code; molecular OBDF at the Γ point; complete and empty active space; sum rule; supercell solvers; second-order quasiparticles of a complete active space | `test_kobdf.py` | branch `thai` (67a4494), internal, `pyscf.fci`, `pyscf.cc` |
| k-point OBMP2: cycle-by-cycle energies of version 0.2; first cycle is KMP2 (also frozen bands, scaled amplitudes); one k-point equals the molecular code; k-point symmetry, k-dependent occupations, integrals on disk, MPI | `test_kobmp2.py` | version 0.2, `pyscf.pbc.mp`, internal |

### Agreement with version 0.2

Largest deviation of total energies from the previous implementation
(commit 29efd04), 6-31G basis unless noted:

| Method | Systems | max \|ΔE\| (Ha) |
| --- | --- | --- |
| OBMP2, O2BMP2 (R, U, exact and DF integrals, BCH 1 and 2, level shift) | H₂O, OH, CH₂ | 1e-10 |
| state-specific OBMP2 / O2BMP2 (MOM) | H₂O, α HOMO→LUMO | 2e-10 |
| OBDF | H₂/cc-pVDZ, LiH, H₂O; 5 active spaces | 4e-10 |
| OBDH, O2BDH | H₂O, CH₂, NH₂ | 4e-11 |
| OBDH, O2BDH | OH (²Π, degenerate) | 1e-7, see `status.md` |
| RO-OBMP2 | OH, CH₂, NH₂ | 1e-13 |
| embedded OBMP2 / OBDH, with and without CL | (H₂O)₂ | 9e-11 |
| KOBMP2 (rewritten, see `kpoints.md`) | diamond, BN; up to 27 k-points | 5e-14 per cycle, 3e-10 converged |

One case of version 0.2 could not be reproduced and is not a test: OBDF for
H₂ at 1.5 Å with the active orbitals [1, 2, 5, 6] (1-based). Orbitals 6 and 7
are degenerate there, so that active space is not uniquely defined; the two
codes differ by 3e-5 Ha and restricted and unrestricted references differ
from each other. `OBDF` now warns in this situation.

### Not validated yet

- **Published numbers.** None of the benchmark values of the papers has been
  recomputed with version 1.0. Suggested set, one per paper: ⟨S²⟩ and
  hyperfine couplings of the radicals of *J. Phys. Chem. A* **125**, 9242;
  dipole moments and density errors of *Phys. Chem. Chem. Phys.* **24**,
  19393; singlet–triplet gaps and ionization potentials of *J. Phys. Chem. A*
  **128**, 1543; charge-transfer excitations of *J. Chem. Phys.* **162**,
  104109; potential curves of *J. Phys. Chem. A* **127**, 5222. The
  geometries and basis sets of those studies are needed for this.
  A first sign of consistency: for CN/cc-pVDZ (`examples/06-properties.py`)
  ⟨S²⟩ drops from 1.150 (UHF) to 0.766 (OBMP2).
- **An independent implementation of the correlation potential.** The tests
  establish that the code reproduces MP2 at the first cycle and the previous
  implementation afterwards. A term-by-term check of $v$ against a direct
  evaluation of the commutators would be an independent validation of the
  second-order terms.
- **OBDH and O2BDH against literature.** Only internal limits are tested.
- **Embedding and k-point downfolding against literature.** Internal limits
  and the previous implementations only.

## Performance

Wall time per cycle, benzene/cc-pVDZ (114 basis functions, 21 occupied
orbitals), 2 threads on a shared virtual machine; variations of about 20 %
between runs. Measure on your own hardware with `benchmarks/timing.py`.

| Calculation | version 0.2 (s) | version 1.0 (s) | ratio |
| --- | --- | --- | --- |
| restricted, exact integrals | 29.5 | 3.0 | 10 |
| unrestricted, exact integrals | 13.4 | 7.6 | 1.8 |
| restricted, density fitting | 10.5 | 1.8 | 6 |
| unrestricted, density fitting | 7.6 | 3.6 | 2.1 |

With density fitting, O2BMP2 costs about 60–70 % of an OBMP2 cycle because the
same-spin blocks are skipped. The remaining cost with exact integrals is the
$O(N^5)$ transformation of the second index pair; use density fitting for
larger molecules.
