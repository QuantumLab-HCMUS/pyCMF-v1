<div align="left">
  <img src="assets/images/PyCMF-logo.jpg" height="90px" alt="pyCMF logo"/>
</div>

# pyCMF

pyCMF is a Python package of **correlated mean-field methods** built on
[PySCF](https://pyscf.org). Its core is one-body second-order Møller–Plesset
perturbation theory (OBMP2): the Hartree–Fock Fock operator is replaced by a
correlated Fock operator that contains a one-body MP2 correlation potential,
and the orbitals, orbital energies and amplitudes are iterated to
self-consistency at the cost of one MP2 calculation per cycle.

| Method | What it is | Entry point | Status |
| --- | --- | --- | --- |
| OBMP2 | self-consistent one-body MP2, restricted and unrestricted | `pycmf.OBMP2` | stable |
| O2BMP2 | OBMP2 with scaled opposite-spin amplitudes only | `pycmf.O2BMP2` | stable |
| OBDH | one-body double hybrid: Kohn–Sham Fock + scaled OBMP2 potential; B2PLYP and other double hybrids by name | `pycmf.OBDH` | stable |
| O2BDH | OBDH with scaled opposite-spin amplitudes only | `pycmf.O2BDH` | stable |
| OBDF | one-body downfolding to an active-space Hamiltonian | `pycmf.OBDF` | stable |
| state-specific excited states | any of the first four with `occupation='mom'` or `'imom'` | keyword | stable |
| RO-OBMP2 | restricted open-shell OBMP2 | `pycmf.mol.roobmp2.ROOBMP2` | experimental |
| embedding | OBMP2- and OBDH-in-DFT projection embedding (separate functionals for subsystem and environment), concentric localization | `pycmf.embed` | experimental |
| k-point OBMP2, O2BMP2 | periodic systems, closed shell: density fitting, k-point symmetry, MPI | `pycmf.pbc.KOBMP2`, `KO2BMP2` | experimental |
| k-point OBDF | downfolding to active bands on the integrals of KOBMP2; mean field, second-order quasiparticles and supercell solvers of the model | `pycmf.pbc.KOBDF` | experimental |
| k-point unrestricted OBMP2 | periodic systems (code of version 0.2) | `pycmf.pbc.KUOBMP2` | experimental |

"Stable" methods are covered by the test suite and exported at the top level.
"Experimental" methods are under development: they are imported from their
sub-package, emit an `ExperimentalWarning`, and are documented in
[docs/status.md](docs/status.md).

## Installation

pyCMF needs Python ≥ 3.9, NumPy, SciPy and PySCF ≥ 2.10.

```bash
git clone https://github.com/QuantumLab-HCMUS/pyCMF-v1.git
cd pyCMF-v1
pip install .            # or: pip install -e ".[dev]" for development
```

## Quick start

```python
from pyscf import gto, scf
import pycmf

mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1)
mf = scf.UHF(mol).run()

mp = pycmf.OBMP2(mf).run()                 # restricted or unrestricted, chosen from mf
print(mp.e_tot, mp.spin_square())

o2 = pycmf.O2BMP2(mf, c_os=1.2).run()      # c_os has no default

mf_df = scf.UHF(mol).density_fit().run()
dh = pycmf.OBDH(mf_df).run()               # B2PLYP by default; xc='PBE0-DH', ...

ham = pycmf.OBDF(mp, ncas=5, nelecas=(3, 2))    # active-space Hamiltonian
e_tot = ham.kernel()                             # FCI in the active space
h1, e_core = ham.get_h1eff()
h2 = ham.get_h2eff()
```

Density fitting is used automatically when the mean-field object uses it, or
on request with `pycmf.OBMP2(mf).density_fit()`. Excited states are computed
state-specifically from a non-Aufbau determinant:

```python
ex = pycmf.O2BMP2(mf_mom, c_os=1.2, occupation='imom').run()
omega = ex.e_tot - o2.e_tot
```

More in [docs/usage.md](docs/usage.md) and in the scripts of
[examples/](examples); the theory is summarised in
[docs/theory.md](docs/theory.md).

## Testing

```bash
pip install ".[test]"
pytest -m "not slow"     # 105 tests, about two minutes on a laptop
pytest                   # adds the slower k-point tests (symmetry, MPI; a few minutes)
```

The tests compare with conventional MP2 and FCI limits from PySCF and with
reference energies of the previous implementation of the methods; see
[docs/validation.md](docs/validation.md). Each example in `examples/` has its
expected output in `examples/reference_output/`.

## Repository layout

```
src/pycmf/
  core/        self-consistent driver, DIIS, occupation schemes (Aufbau/MOM), amplitudes
  mol/         molecules: integrals, correlation potential, OBMP2, O2BMP2, OBDH, O2BDH, RO-OBMP2
  downfold/    one-body downfolding, FCIDUMP export
  properties/  densities, dipole moments, <S^2>, hyperfine couplings
  embed/       projection embedding and concentric localization   (experimental)
  pbc/         k-point OBMP2, O2BMP2 and OBDF                       (experimental)
tests/         pytest suite
examples/      runnable scripts with reference output
benchmarks/    timing script; legacy inputs of version 0.2
docs/          theory, usage, validation, status, migration from version 0.2
paper/         material for the Computer Physics Communications submission
```

## Development

### Environment

If `PYTHONPATH` points at another PySCF checkout (for example an older fork), it shadows the PySCF installed in
the virtual environment. Install into a venv and run everything with `PYTHONPATH` unset:

```bash
python3 -m venv .venv
env -u PYTHONPATH .venv/bin/pip install -e ".[dev]"     # dev adds ruff + pre-commit; ".[test]" is pytest only
env -u PYTHONPATH .venv/bin/python -m pytest -m "not slow"   # required before merging
env -u PYTHONPATH .venv/bin/python -m pytest                 # + 9 slow k-point tests (symmetry, larger meshes, MPI)
env -u PYTHONPATH .venv/bin/python -m pytest tests/test_obmp2.py::test_name   # single test
env -u PYTHONPATH .venv/bin/ruff check src/pycmf tests examples              # lint, as in CI
cd examples && env -u PYTHONPATH ../.venv/bin/python check_examples.py [--all]  # examples vs reference_output/
```

`pytest.ini` sets `pythonpath = src`. Ruff uses line length 110 and single quotes, and excludes `src/pycmf/pbc`
and `benchmarks`. The pre-commit hook runs only `ruff format`. CI (`.github/workflows/ci.yml`) runs on Python 3.9
and 3.12.

### Architecture

**One self-consistent driver for every method.** `core/scf_loop.kernel(mp, mo_energy, mo_coeff, nocc)` iterates
over a list of *channels*:
- a channel is a spin channel for molecules and an irreducible k-point for `pbc`;
- each cycle calls `mp.build(...)` and gets back an `Intermediates` object (`e_tot`, `fock_ao`, `fock_mo`,
  `dm_chan`, ...);
- it then applies DIIS on the commutator [F, D], diagonalizes in the span of the current orbitals, and occupies the
  orbitals through `core/occupation.select`. The schemes are `aufbau`, `mom` and `imom`.
- Convergence requires both |ΔE| < `conv_tol` and |f_ov| < `conv_tol_grad`.

A new method mostly means supplying its own `build`.

**Molecular OBMP2 (`mol/obmp2.py`).**
- `OBMP2Base(lib.StreamObject)` has two subclasses: `ROBMP2` (closed shell, a single channel with `_occ = 2`) and
  `UOBMP2` (two channels). `_pack`/`_unpack` convert between PySCF arrays and the channel lists.
- The factory `OBMP2(mf)` picks the subclass from the type of `mf`. `O2BMP2` is `OBMP2` with `c_ss=0` and requires
  an explicit `c_os`.
- `build` works in this order:
  1. HF Fock matrix from the current density;
  2. integrals: `mol/integrals.py`, `IncoreInts` or `DFInts` depending on `with_df`;
  3. amplitude blocks: `core/amplitudes.make_blocks`, which applies the `c_os`/`c_ss` scaling to the amplitudes;
  4. correlation potential: `mol/bch.correlation_potential`, controlled by `bch_order` (1 or 2, default 2);
  5. `get_reference(...)` returns the reference Fock matrix, the reference energy and the weight of v.
- Variants are hooks on this structure, not new modules:
  - `mol/obdh.py` (`OBDH`, `O2BDH`): a mixin overrides `get_reference` to use a Kohn–Sham Fock matrix and scales v
    by `a_c`. Double hybrids are parsed by name (B2PLYP is the default).
  - `mol/roobmp2.py`: changes the orbital update instead.
- `downfold/obdf.py` (`OBDF`) builds an active-space Hamiltonian from a converged OBMP2 object.
  `h̄ = h + v_core^HF + v[T] − v[T_act]`, where `restrict_to_active` gives `T_act`.
- `properties/` (rdm, spin, hfcc) uses the stored intermediates (`get_intermediates`).

**Periodic code (`pbc/`, experimental).**
- `KOBMP2`/`KO2BMP2` (`kobmp2.py`) reuse the same `scf_loop`, with the irreducible k-points as channels.
- The supporting modules are:
  - `kintegrals.py`: three-index GDF tensors, kept in memory or in HDF5;
  - `kbch.py`: correlation potential, never forming integrals with a general index;
  - `ksymm.py`: space-group symmetrization;
  - `core/mpi.py`: MPI over ki.
- Built on top of KOBMP2: `kipea.py`, `kqp.py` (quasiparticles) and `kpeom.py` (P-EOM-MP2).
  `kobdf.py` (`KOBDF`) does downfolding using only these pieces.
- `kuobmp2.py` is the 0.2 four-index code, kept largely as is; see the known issues in `docs/status.md`.

**Maturity.** Stable methods are exported in `pycmf/__init__.py`. Experimental ones (RO-OBMP2, `embed/`, `pbc/`)
stay out of the top-level exports and call `core.status.experimental(name)` in `__init__`, which raises an
`ExperimentalWarning`. Promoting a method means:
1. validate it against published or independent results;
2. export it;
3. update `docs/status.md`.

### Conventions (from CONTRIBUTING.md)

- **One method per module.** DF, DIIS, MOM, spin scaling and BCH order are attributes of the method object. Never
  add modules named `*_slow`, `*_fast`, `*_mom`, `*_new`, and so on.
- **Placement.** Molecular code goes in `mol/`, periodic code in `pbc/`, and code shared by both in `core/`.
  Reference implementations written as explicit loops belong in `tests/`, not in the package.
- **PySCF style.**
  - `lib.StreamObject`, options as attributes, `kernel()` returns the main result (`e_tot`);
  - `pyscf.lib.logger`, never `print`;
  - **never modify the user's mean-field object** (there is a test for this).
- **Tests for a new method.** Add three kinds:
  - an exact limit, for example "first cycle from HF orbitals = PySCF MP2", "all orbitals active = FCI" or
    "a_c = 0 is hybrid DFT";
  - a regression value;
  - input checks.

  `tests/conftest.py` provides the session-scoped `mf_cache` fixture with shared molecules. Regression energies
  come from pyCMF 0.2 (commit 29efd04) and use a tolerance of about 5e-9 Ha.
- **Branches.** `feature/<name>` branches are short-lived and rebased on the main branch. Releases are tags.

### Docs

| File | Contents |
| --- | --- |
| `docs/theory.md` | equations |
| `docs/usage.md` | API |
| `docs/status.md` | maturity table and known issues |
| `docs/validation.md` | what each test checks |
| `docs/kpoints.md` | k-point algorithm, timings, `KOBDF` |
| `docs/migration.md` | name mapping from 0.2 to 1.0 |

If you change behaviour, numbers or test counts, update the matching doc. The old 0.2 inputs in
`benchmarks/legacy-0.2/` use the old interface and do not run as is.

## Citing

If you use pyCMF, please cite the papers of the methods you use
(see [CITATION.cff](CITATION.cff) and [docs/theory.md](docs/theory.md)):

- OBMP2: T. N. Lan and T. Yanai, *J. Chem. Phys.* **138**, 224108 (2013);
  L. N. Tran, *J. Phys. Chem. A* **125**, 9242 (2021)
- densities: L. N. Tran, *Phys. Chem. Chem. Phys.* **24**, 19393 (2022)
- O2BMP2: N. T. Tran, H. T. Nguyen and L. N. Tran, *J. Phys. Chem. A* **128**, 1543 (2024)
- state-specific excited states: N. T. Tran and L. N. Tran, *J. Chem. Phys.* **162**, 104109 (2025)
- downfolding: N. T. Le and L. N. Tran, *J. Phys. Chem. A* **127**, 5222 (2023)

and PySCF: Q. Sun *et al.*, *WIREs Comput. Mol. Sci.* **8**, e1340 (2018).

## License

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).

pyCMF is developed by QuantumLab HCMUS, University of Science, Vietnam
National University Ho Chi Minh City.
