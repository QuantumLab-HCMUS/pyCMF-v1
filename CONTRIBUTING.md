# Contributing to pyCMF

## Layout rules

* **One method, one module.** Algorithmic variants (density fitting, DIIS, MOM,
  spin scaling, BCH order) are attributes of the method object, not new files.
  Do not add modules with suffixes such as `_new`, `_fast`, `_slow`, `_mom`.
* **By kind of system.** Molecular code lives in `pycmf/mol`, periodic code in
  `pycmf/pbc`. Code that is the same for both (self-consistent driver, DIIS,
  occupation schemes, amplitude scaling) belongs in `pycmf/core`.
* **Reference implementations** (explicit loops, for checking) belong in
  `tests/`, not in the package.

## Adding a method

1. Put it at its final location and call `pycmf.core.status.experimental(name)`
   in `__init__`. Do not export it in `pycmf/__init__.py`.
2. Reuse `core.scf_loop.kernel` by providing `build(mo_energy, mo_coeff, nocc)`;
   see `mol/obdh.py` (a different reference Fock matrix) and `mol/roobmp2.py`
   (a different orbital update) for the two usual cases.
3. Add tests: a limit that must hold exactly (for example the first cycle
   against PySCF), a regression value, and input checks.
4. When the method is validated against published or independent results,
   remove the experimental marker, export it, and move its entry in
   `docs/status.md`.

## Style

* PySCF conventions: `lib.StreamObject`, a module-level `kernel`, attributes
  for options, `kernel()` returns the main result, `pyscf.lib.logger` instead
  of `print`.
* Never modify the mean-field object passed by the user.
* `ruff check src/pycmf tests examples` must pass (line length 110, single quotes).

## Tests

```bash
pytest -m "not slow"     # required before every merge
pytest                   # includes the k-point test
```

`master` must always pass the tests. Develop on short-lived `feature/<name>`
branches and rebase on `master` regularly; releases are tags.
