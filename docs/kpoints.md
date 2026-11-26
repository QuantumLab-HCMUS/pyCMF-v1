# k-point OBMP2: algorithm, usage and performance

`pycmf.pbc.KOBMP2` (and `KO2BMP2`) is a rewrite of the k-point code of
version 0.2. It is still **experimental**: it reproduces the previous
implementation and PySCF's KMP2, but has not been compared with published
numbers. This page records what was changed and what was measured.

## Usage

```python
from pyscf.pbc import gto, scf
from pycmf.pbc import KOBMP2, KO2BMP2

mf = scf.KRHF(cell, cell.make_kpts([4, 4, 4])).density_fit().run()
mp = KOBMP2(mf).run()            # e_tot, e_corr per unit cell
mp.band_gap()                    # from the bands of the correlated Fock matrix
```

| Attribute | Default | Meaning |
| --- | --- | --- |
| `conv_tol`, `conv_tol_grad`, `max_cycle` | 1e-8, √conv_tol, 100 | convergence on the energy and on the occupied–virtual block of the correlated Fock matrix |
| `diis`, `diis_space`, `diis_start_cycle` | True, 8, 1 | DIIS on the Fock matrices of the irreducible k-points |
| `bch_order` | 2 | 1: half-weighted first commutator, as for molecules |
| `c_os`, `c_ss` | 1, 1 | scaling of the amplitudes; `KO2BMP2(mf, c_os=...)` sets `c_ss = 0` |
| `level_shift` | 0 | shift of the energy denominators |
| `frozen` | None | number of lowest bands, or list of band indices, without amplitudes; must not cut a degenerate level (in diamond the two lowest valence bands are degenerate at X) |
| `kpt_symmetry` | True | True: space group and time reversal; `'time_reversal'`; False |
| `with_df` | from the mean field | GDF or RSGDF object; `mp.density_fit(auxbasis=...)` builds another one |
| `max_memory` | from the mean field | decides whether three-index integrals stay in memory |
| `direct_exchange` | True | take the exchange matrices of the Hartree–Fock part from the pass over the three-index integrals that builds the amplitudes (plain `KRHF` on the same GDF object, no MPI); identical energies, one pass over the integral file less per cycle |
| `single_precision` | False | keep the occupied–virtual three-index tensors in single precision (half the memory; the arithmetic stays in double precision; energy change below 1e-9 Ha for diamond gth-dzvp 3×3×3) |

Restart: set `mp.mo_coeff`, `mp.mo_energy` and `mp.mo_occ` (lists over the
k-points of the mesh, e.g. from a converged calculation with other scaling
factors) before `kernel()`.

Disk: the three-index file of PySCF holds all pairs of k-points; for diamond
gth-dzvp it takes 3.9 GB at 4×4×4 and 14.8 GB at 5×5×5, and PySCF needs
about as much again for a temporary file while it is built. Set
`mf.with_df._cderi_to_save` to keep the file and `mf.with_df._cderi` to
reuse it.

Results: `e_tot`, `e_corr`, `e_ref`, `converged`, `mo_energy`, `mo_coeff`,
`mo_occ` for all k-points of the mesh, `get_fock()`, `ip_koopmans()`,
`ea_koopmans()`, `band_gap()`. The old attribute names `niter`, `thresh`,
`shift` and `ene_tot` are kept as aliases.

### Quasiparticle gaps (IPEA)

The eigenvalues of the correlated Fock matrix are not removal and addition
energies; `band_gap()` is their gap. `mp.ipea(nvir=4)` adds the second-order
corrections

    IP(h,k) = −ε(h,k) + 2 Σ t̄_iajh (ia|jh),   EA(l,k) = −ε(l,k) − 2 Σ t̄_ialb (ia|lb)

to the converged eigenvalues (`pbc/kipea.py`), stores `mp.ip` and `mp.ea` for
all k-points of the mesh and returns the quasiparticle gap min IP − max EA.
The amplitudes carry the same `c_os`, `c_ss` scaling as the potential. The
corrections are static and are evaluated once; only the band edges are
meaningful, because the denominators can vanish for bands away from the gap.
The cost is negligible compared with one self-consistent cycle.

The routine `make_IPEA` of version 0.2 evaluated these terms at Γ only, for
one band each; its EA is reproduced to 1e-9 Ha. Its IP read the exchange
integrals of k-point pairs that the loop had not yet filled, so that part of
the exchange term was missing (3 mHa for diamond gth-szv 2×2×2); the new
routine agrees with a direct four-index evaluation to 1e-8 Ha.

`benchmarks/diamond_gaps.py` computes the gaps of diamond for HF, second-order
quasiparticle energies on HF, KOBMP2, SCS- and SOS-KOBMP2 with the static
(IPEA), linearized and dynamic corrections on one mesh. With `exxdiv=None`,
GTH-DZVP and 3×3×3 k-points it gives 12.22, 2.76, 11.50 and 3.67 eV for
KOBMP2, IPEA-KOBMP2, SCS-KOBMP2 (c_os = 1.2, c_ss = 1/3) and
IPEA-SCS-KOBMP2, within about 0.1 eV of the values obtained with the 0.2
code and FFT-based integrals (12.12, 2.64, 11.40, 3.69 eV). These gaps
depend strongly on the mesh and on the treatment of the exchange divergence;
both `exxdiv=None` and `exxdiv='ewald'` series must be extrapolated, with
a + b Nk^(-1/3) + c Nk^(-2/3), before they can be compared with anything.

### Quasiparticle energies beyond the static correction

`pbc/kqp.py` tabulates the two parts of the diagonal second-order
self-energy on a frequency grid around each band energy,

    A_p(ω) = Σ_iab W / (ω + ε_i − ε_a − ε_b)   (2p1h),
    B_p(ω) = Σ_iaj W / (ω + ε_a − ε_i − ε_j)   (2h1p).

With Hartree–Fock orbitals the diagonal of the OBMP2 potential is exactly one
of them: v_hh = A_h(ε_h) for occupied and v_ll = B_l(ε_l) for virtual bands
(tested to 1e-8 Ha). The eigenvalues of the correlated Fock matrix therefore
contain the half of the static self-energy that opens the gap, and the IPEA
correction is the other half, B_h and A_l. Nothing is counted twice.

    mp.kernel()
    mp.self_energy(nvir=1, nocc=1)             # band edges only
    direct, smallest = mp.quasiparticle_gap('dynamic')

| scheme | quasiparticle energy |
|---|---|
| `'static'` | ε_p + Σ_out(ε_p); identical to `mp.ipea()` |
| `'linear'` | ε_p + Z Σ_out(ε_p), Z = 1/(1 − A′ − B′) |
| `'dynamic'` | root of ω = ε_p + [Σ_in(ω) − Σ_in(ε_p)] + Σ_out(ω) |

`reference='hf'` together with `KOBMP2(mf).init_from_scf()` gives the
conventional second-order quasiparticle energies on Hartree–Fock,
ε_p + A + B.

Spin-component scaling. The amplitudes enter the potential linearly with
the factor c and quadratically with c², so the half inside the eigenvalues
carries 2c − c² per spin component (0.96 and 0.56 for SCS with 1.2 and 1/3;
0.91 for SOS with 1.3), whereas the IPEA formula multiplies the other half by
c itself. `self_energy(scaling='hylleraas')` evaluates the self-energy with
2c − c², so that both halves are weighted alike; the default `'linear'`
reproduces the IPEA numbers. For c_os = c_ss = 1 there is no difference.

Feeding the quasiparticle energies back into the denominators (eigenvalue
self-consistency) was tried and lowers the gaps, because the denominators
already contain the large gap of the OBMP2 eigenvalues.

### Partitioned EOM-MP2 (dressed self-energy)

`pbc/kpeom.py` computes the band edges with partitioned EOM-MP2: the
eigenvalues of exp(−T2) H exp(T2) in the space of 1h and 2h1p (1p and 2p1h)
configurations, with the doubles–doubles block replaced by orbital-energy
differences. Eliminating the doubles gives the same structure as the
second-order self-energy above, with one of the two integrals of every term
dressed by the amplitudes; the dressing contains the leading screening (ring),
ladder and exchange terms of third order.

    mp = KOBMP2(mf).run()                 # or KOBMP2(mf).init_from_scf() for HF orbitals
    res = mp.peom_mp2(kpt_v=0, kpts_c=[0, kc])
    res['gap'][kc]                        # Hartree

Orbitals and orbital energies (amplitude denominators, doubles–doubles block)
are those of the KOBMP2 object; the singles block and the couplings use the
Fock matrix of Hartree–Fock in that basis. Only the target k-point is an
external index, so no array with three k-point indices is stored, and the
k-points that are equivalent under the operations leaving the target k-point
fixed are computed once. With `symmetry=True` (default) the singles space is
the degenerate level of the band edge; the neglected coupling to the other
bands of the k-point changes the electron affinity of diamond by 0.01–0.03 eV.

Validation: IP and EA agree with the EOM-KCCSD code of PySCF (t1 = 0,
MP2-like t2, the same doubles–doubles block) to 1e-9 Ha for a distorted
diamond cell on a shifted mesh, with HF and with OBMP2 orbitals
(`tests/test_kobmp2.py`); the PySCF route stores Nk³ arrays with three
virtual indices and is limited to very small meshes.

Results (GTH-PADE, gth-dzvp, meshes 2×2×2 to 4×4×4 with both `exxdiv`,
extrapolated; indirect gaps in eV): P-EOM-MP2 on KOBMP2 orbitals gives 6.1
(diamond), 6.5 (c-BN) and 2.7 (Si); on HF orbitals 6.1, 6.2 and 2.9. The
static second-order correction (IPEA-KOBMP2) gives −1.9, −0.1 and −6.3, its
dynamic solution 2.0, 3.2 and −1.8. Experiment without zero-point
renormalization: 5.81, 6.61, 1.30. The extrapolation is uncertain by about
0.5 eV for the static scheme and, for P-EOM-MP2, 0.2 eV (diamond) to 0.4 eV
(Si). A 5×5×5 point (`exxdiv=None`, HF orbitals) lies within 0.03 eV of the
value predicted by the 2×2×2–4×4×4 fit for diamond and 0.1 eV above it for
Si; it leaves both extrapolated gaps unchanged but does not narrow the
spread between fit forms for Si.

### Nearly linearly dependent basis sets

The OBMP2 iterations are less robust than Hartree–Fock when the Bloch basis
is nearly linearly dependent. For diamond with gth-tzvp PySCF removes two to
four functions per k-point, but overlap eigenvalues below 1e-4 remain, and
with them KOBMP2 does not converge (|f_ov| stays near 1e-2; without DIIS it
diverges). `remove_linear_dependency_(mf, 1e-4)` (from `pycmf.pbc`, before
`mf.kernel()`) makes the mean field work in the space of overlap eigenvectors
above the threshold; KOBMP2 keeps that orbital space and converges in five to
six cycles, and the Hartree–Fock energy changes by 5e-5 Hartree. The number of
orbitals then depends on the k-point, the mean field is no longer exactly
symmetric, and KOBMP2 falls back to the full mesh.

Restrictions: closed-shell `KRHF` reference with integer occupations (the
number of occupied bands may depend on the k-point); three-dimensional
systems; Gaussian density fitting for the correlation part. If the mean
field uses FFT-based integrals, a `GDF` object is built for the correlation
part, so the result differs from an all-FFT calculation by the density
fitting error.

### MPI

With `mpi4py` the loop over k-points of the correlation potential is
distributed; the same script runs on every process and rank 0 does the
Hartree–Fock part. See `examples/experimental/kobmp2_mpi.py`:

```bash
OMP_NUM_THREADS=1 mpiexec -n 4 python kobmp2_mpi.py
```

Processes times threads must not exceed the number of cores: oversubscribed
BLAS threads slow the calculation down by orders of magnitude.

## What was wrong with the 0.2 code

Measured for diamond (2 atoms, GDF, 2 threads):

1. `tracemalloc.start(25)` was called in every cycle and never stopped. It
   slows down every allocation of the process: 53–88 s instead of 9.6 s for
   two cycles with 2×2×2 k-points.
2. Every cycle called `with_df.ao2mo` once per k-point triple, for a full
   (ov|gg) block: Nk³ four-index transformations, 70–90 % of the remaining
   time, and the reason for the Nk³ growth of the prefactor (37 times more
   time for 27 instead of 8 k-points).
3. `tmp_bar[Nk, Nk, o, v, o, v]` and three copies of the integrals were held
   in memory.
4. A space-group object was built in every cycle and not used; the core
   Hamiltonian was recomputed in every cycle; the ionization/affinity
   routine with hard-coded band indices ran at the end of every calculation.
5. No DIIS, convergence on the energy only.
6. The orbitals of the mean-field object were overwritten in place, so that
   a second calculation with the same `mf` started from the orbitals of the
   first one.

## Algorithm

**Three-index integrals.** All integrals are products of GDF tensors,
(ia|jb) = Σ_Q L^Q_ia(ki,ka) L^Q_jb(kj,kb). The occupied–virtual tensors
("rows", one per k-point) are transformed once per cycle and kept in memory
or, when they need more than half of `max_memory`, in a temporary HDF5 file.
The AO tensors are cached between cycles as far as the remaining memory
allows (up to 80 % of `max_memory` in total).

**No integral with a general index.** The terms Σ t̄_iajb (ia|pb) and
Σ t̄_iajb (ia|jp) are evaluated through
N^Q_jb(kb) = Σ_ki Σ_ia L^Q_ia t̄_iajb, accumulated over ki, and contracted
once per pair (kj, kb) with L^Q_pb and L^Q_jp, which are built on the fly.

**Memory.** Nothing of size Nk²·o²v² is stored; the amplitudes exist for one
k-point quadruple at a time. The singles-like term that needed `tmp_bar` is
evaluated in a second pass over the Nk·N_ibz momentum-diagonal amplitudes.

**Matrix products.** All contractions of a quadruple are products of
reshaped arrays (one BLAS call each for large blocks, little interpreter
overhead for small ones); the exchange partner (ki, kb, kj, ka) of a
quadruple reuses its integrals.

**Symmetry.** Every term of the potential is written with its free indices
on kj ("j terms") or on kb ("b terms"), using t_iajb = t_jbia for the terms
that 0.2 attached to ki and ka. kj then runs over the irreducible k-points
only: the j terms are complete, and the b terms are completed by the group
average v = 1/|G| Σ_g g⁻¹ P g of the partial sum P, with the orbital
rotations of `pyscf.pbc.symm`. The orbitals are kept at the irreducible
k-points and expanded in a symmetry-consistent gauge in every cycle. If the
mean-field solution does not have the symmetry of the mesh, the full mesh is
used.

**Self-consistency.** The loop is `pycmf.core.scf_loop`, with the
irreducible k-points as channels: DIIS with the commutator [F, D],
convergence on ΔE and on the occupied–virtual block of the correlated Fock
matrix, diagonalization in the space of the current orbitals (so that
orbitals removed because of linear dependencies stay removed).

**Parallelism.** BLAS threads inside the products; MPI over ki.

## Validation

| Check | Systems | Largest deviation (Ha) |
| --- | --- | --- |
| 0.2 implementation (four-index, GDF), cycles 1, 3 and 6 without DIIS (fewer for the larger cases), no symmetry | diamond szv 1×1×2, 2×2×2, 3×3×3, shifted 2×2×2; diamond dzvp 2×2×2; distorted diamond 2×2×1; BN 2×2×2 | 5e-14 |
| same, converged energy and band energies | diamond 1×1×2, 2×2×2; BN 2×2×2 | 3e-10 (energy), 3e-8 (bands) |
| with k-point symmetry against the 0.2 implementation | diamond 2×2×2 (3 of 8 k-points), 3×3×3 (4 of 27), dzvp 2×2×2; BN 2×2×2 | 2e-10 |
| three symmetry modes against each other, converged | BN 3×3×3 (27 / 14 / 4 k-points) | 1e-11 |
| first cycle = E(KRHF) + E(KMP2), also with a frozen band and frozen virtuals | diamond, BN | 2e-10 |
| first cycle with scaled amplitudes = Hylleraas value (2c − c²) of the KMP2 components | BN 2×2×2 | 1e-10 |
| one k-point = molecular code with the same integrals: default, O2BMP2, SCS-type scaling, level shift, `bch_order=1` | BN, Γ point | 4e-10 |
| k-dependent occupations: first cycle = KMP2; symmetry on/off | hydrogen lattice 2×2×2 | 3e-11 |
| three-index tensors on disk = in memory | diamond, BN | 2e-15 |
| 2 MPI processes = serial (OBMP2, frozen band, O2BMP2; symmetry on/off) | BN 3×3×3 | 4e-15 |
| symmetry-adapted KRHF as input = full-mesh KRHF | BN 2×2×2 | 5e-13 |

The 0.2 code updates the orbitals of `mf` in place; its runs with 1, 2 and 3
cycles from the same `mf` therefore end at cycles 1, 3 and 6 of one
calculation, which is how the reference values are labelled.

## Timings

Diamond, two atoms per cell, GTH pseudopotential, GDF, 2 threads of a shared
virtual machine (timings fluctuate by about ±20 %). Time per self-consistent
cycle, including the Hartree–Fock Fock build of PySCF:

| Basis, mesh | k-points (irreducible) | 0.2, debugging overhead removed | 1.0, full mesh | 1.0, symmetry | Peak memory |
| --- | --- | --- | --- | --- | --- |
| gth-szv, 2×2×2 | 8 (3) | 4.2 s | 0.30 s | 0.17 s | 0.2 GB |
| gth-szv, 3×3×3 | 27 (4) | 149 s | 3.7 s | 2.1 s | 0.3 GB |
| gth-dzvp, 2×2×2 | 8 (3) | 11 s | 1.1 s | 0.8 s | 0.3 GB |
| gth-szv, 4×4×4 | 64 (8) | ≈ 36 min (extrapolated with Nk³) | 33 s | 11.5 s | 0.9 GB |
| gth-dzvp, 3×3×3 | 27 (4) | ≈ 7 min (extrapolated with Nk³) | 27 s | 7.5 s | 1.7 GB |
| gth-dzvp, 4×4×4 | 64 (8) | ≈ 94 min (extrapolated with Nk³) | not run | 88 s | 3.5 GB |

The 0.2 code as it was (with `tracemalloc`) is another factor of 5–9 slower
than the third column. With symmetry the correlation potential is no longer
the largest part of a cycle for these sizes: for 3×3×3 gth-szv it takes
1.0 s of 3.4 s, the rest is PySCF's Hartree–Fock build. For 4×4×4 gth-dzvp
(88 s) the parts are: amplitudes and their contractions 33 s, reading the AO
three-index integrals from the GDF file 20 s, Hartree–Fock build 19 s,
transformations 14 s. The reading disappears when `max_memory` is large
enough to keep all AO tensors (7.4 GB for this case; the default of
4000 MB holds a third of them).

Cycles to |ΔE| < 1e-8 Ha from Hartree–Fock orbitals:

| Basis, mesh | with DIIS | without |
| --- | --- | --- |
| gth-szv, 2×2×2 | 5 | 7 |
| gth-szv, 3×3×3 | 7 | 12 |
| gth-dzvp, 2×2×2 | 7 | 12 |
| gth-szv, 4×4×4 | 7 (104 s in total) | 13 (179 s) |

MPI, BN 3×3×3 gth-szv, three cycles, one thread per process: 21.6 s with one
process and 15.9 s with two on the full mesh; 10.5 s and 8.2 s with symmetry.
For this small case the serial parts (Hartree–Fock build, reading of the
integrals) limit the gain.

## Downfolding (`KOBDF`)

`KOBDF(mp, ncas, nelecas)` (or `mp.downfold(ncas, nelecas)`) builds an
active-space Hamiltonian from a `KOBMP2` object, per unit cell

$$
\hat H_{\rm act}=E_{\rm core}+\frac1{N_k}\sum_{k}\sum_{tu}\bar h_{tu}(k)\,
\hat a^\dagger_{tk}\hat a_{uk}+\frac1{2N_k}\sum_{k_1k_2k_3}\sum_{tuvw}
h^{(2)}_{tuvw}(k_1,k_2,k_3)\,\hat a^\dagger_{tk_1}\hat a^\dagger_{vk_3}\hat a_{wk_4}\hat a_{uk_2},
$$

$$
\bar h(k)=h(k)+v^{\rm HF}_{\rm core}(k)+v[T](k)-v[T_{\rm act}](k),\qquad
h^{(2)}=(tk_1,uk_2|vk_3,wk_4)/N_k .
$$

It uses the machinery of `KOBMP2` and nothing else:

- $v[T]$ and $v[T_{\rm act}]$ are two calls of
  `kbch.correlation_potential`, with all correlated orbitals and with the
  active ones (the routine that implements frozen bands), on the irreducible
  k-points.
- All two-electron integrals are products of the three-index tensors of
  `kintegrals`, $L^Q_{tu}(k_1,k_2)$ for the active orbitals
  (`get_cderi`). The four-index array of the mesh is assembled only on
  request (`get_h2eff()`).
- The second-order self-energy of the model (`self_energy`,
  `quasiparticle`) is evaluated by the routine of `kqp` that serves
  `KOBMP2.self_energy`.

The cost is one KOBMP2 cycle plus one cycle restricted to the active bands.

**Mean field and quasiparticles of the model.** `scf()` solves Hartree–Fock
for $\hat H_{\rm act}$. Its bands contain the correlation with the orbitals
outside the active space statically, through $v^{\rm ext}$.
`self_energy()` adds the correlation within the active space with its
frequency dependence (2p1h and 2h1p parts), and `quasiparticle('static' |
'linear' | 'dynamic')` solves for the bands as on the Hartree–Fock reference
in `kqp`. For Hartree–Fock orbitals and a complete active space this is the
conventional second-order quasiparticle energy (a test).

**Many-body solvers.** `supercell()` returns real integrals of the
Born–von Kármán supercell of $N_k$ cells ($N_k n_{\rm cas}$ orbitals). The
real orbitals are obtained from time reversal: complex conjugation maps the
active Bloch orbitals of $k$ on those of $-k$,
$\psi_k^*=\psi_{-k}T(k)$; the matrix $T$ of all k-points is unitary and
symmetric, $T=VV^{\rm T}$, and $\psi V$ is real. The orbitals are not
localized. `as_scf()` wraps the integrals in a PySCF `RHF` object,
`kernel()` runs FCI and returns the energy per cell.

**Checks** (`tests/test_kobdf.py`): the four-index KROBDF code of the group
(constant, one-body integrals and mean-field bands to 1e-8 Ha, `exxdiv=None`);
the molecular `OBDF` at the Γ point (constant, integrals and FCI energy to
1e-8 Ha); complete active space ($\hat H_{\rm act}=\hat H$, Fock matrix of
KRHF) and empty one ($E_{\rm core}=E_{\rm KOBMP2}$), with and without the
Ewald treatment of the exchange; the sum rule
$\langle\Phi|\hat H_{\rm act}|\Phi\rangle=E_{\rm KOBMP2}-E_c[T_{\rm act}]$;
k-point symmetry on and off; supercell RHF equal to `scf()`, CCSD between
Hartree–Fock and FCI.

**Madelung term.** With `exxdiv='ewald'` the constant contains
$-\xi N_{\rm act}/2$ for the active electrons and `get_fock` shifts the
occupied levels by $-\xi$, as PySCF does for KRHF; the integrals do not
contain the term. See [status.md](status.md) for the difference to KROBDF.

## Not done

- No GPU path: there was no GPU to test one on.
- No Laplace-transformed O(N⁴) algorithm for KO2BMP2; opposite-spin scaling
  only skips the exchange integrals. With density fitting and eight
  quadrature points the Laplace route costs naux²·o·v·ngrid (first order) and
  naux²·o·v·ngrid² (second order) per k-point pair against naux·o²v² and
  o²v³ now: it pays for more than about 30 (first order) and several hundred
  (second order) occupied orbitals per cell, and not at all for the small
  cells with k-point symmetry this code is used for.
- P-EOM-MP2 keeps only the degenerate frontier level in the singles space
  and is not MPI-parallel; no ring-CCD amplitudes.
- No real arithmetic for Γ-only or inversion-symmetric meshes.
- Only symmorphic space-group operations are used (PySCF default).
- `KUOBMP2` is still the four-index code of 0.2.
- `KOBDF`: closed shell only; no localization of the supercell orbitals; the
  Wannier90, band and DOS tools of the KROBDF package are not ported.
- No finite-size correction and no comparison with published band gaps or
  lattice energies.
