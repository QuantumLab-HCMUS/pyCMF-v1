# Benchmarks

* `timing.py` — wall time per OBMP2 cycle for benzene (restricted/unrestricted,
  exact/density-fitted integrals).
* `kpoints_timing.py` — wall time per cycle of `KOBMP2` for diamond and BN.
* `diamond_gaps.py` — band gaps of diamond on one k-mesh: HF, second-order
  quasiparticle energies on HF, KOBMP2, SCS- and SOS-KOBMP2 with static (IPEA),
  linearized and dynamic corrections.
* `laplace_o2bmp2.py` — error of Laplace-transformed energy denominators in
  self-consistent O2BMP2 as a function of the number of quadrature points.
* `legacy-0.2/` — input scripts of pyCMF 0.2. They use the old interface
  (see `docs/migration.md`) and are kept as a record of the systems studied;
  they do not run with version 1.0.
