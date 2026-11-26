# Theory and working equations

This page summarises what pyCMF computes, in the notation of the code. The
derivations are in the papers listed at the end.

## 1. OBMP2

The electronic Hamiltonian is transformed with an anti-Hermitian doubles
operator $\hat A$ built from first-order amplitudes, the
Baker–Campbell–Hausdorff expansion is truncated at second order (with the
Fock operator $\hat F$ in the double commutator), and every commutator is
reduced to one-body operators and constants by a cumulant approximation,
$[\ldots]_1$:

$$
\hat H_{\rm OBMP2} = \hat H_{\rm HF} + \big[\hat H,\hat A\big]_1
 + \tfrac12\big[[\hat F,\hat A],\hat A\big]_1 = \hat{\bar F} + \bar C ,
\qquad \bar f_{pq} = f_{pq} + v_{pq}.
$$

$v$ is the **correlation potential** and $\bar f$ the **correlated Fock
matrix**. The orbitals are the eigenvectors of $\bar f$ and the orbital
energies $\varepsilon$ its eigenvalues. The amplitudes are rebuilt from them,

$$
t^{ab}_{ij} = c\,\frac{(ia|jb)}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b-\Delta},
$$

and the procedure is iterated until the energy is stable and the Brillouin-type
condition $\bar f_{ia}=0$ holds. $c=1$ in OBMP2, $\Delta$ is an optional
level shift (`level_shift`). Starting from HF orbitals, the first cycle is
conventional MP2.

The energy of a cycle is

$$
E = E_{\rm HF}[\Phi] + c_0 + \sum_{i}^{\rm occ} v_{ii},
$$

where $E_{\rm HF}[\Phi]$ is the HF energy of the current determinant and
$c_0$ the constant of the correlation potential. Because the amplitudes
depend on the orbital energies, this functional is not stationary with
respect to orbital rotations; the converged orbitals are defined by the
eigenvalue problem, not by a variational principle.

### Spin blocks

Amplitudes are stored in chemists' order, `t[i,a,j,b]`, as a list of spin
blocks $(x,y)$: the pair $(i,a)$ has spin $x$, the pair $(j,b)$ spin $y$.
$\bar t = t - t^{\rm exch}$ for same-spin blocks, $\bar t = t$ for
opposite-spin blocks. An unrestricted reference has the four blocks
$\alpha\alpha$, $\beta\beta$, $\alpha\beta$, $\beta\alpha$. A restricted
reference has one spin channel and two blocks (same spin, opposite spin); the
spin-flipped partners contribute a factor of two to the constants. The same
routines (`pycmf.mol.bch`) therefore serve both cases.

### Working equations

With $f$ the HF Fock matrix in the current orbital basis and sums over
repeated indices, each block $(x,y)$ adds to the unsymmetrised matrices
$c_1^{x}$, $c_1^{y}$ and to the constant $c_0$.

First order, $[\hat H,\hat A]_1$:

$$
\begin{aligned}
c_1^{y}[p,k] &\mathrel{+}= 2\,(ia|pl)\,\bar t_{ia,kl}, &
c_1^{y}[p,l] &\mathrel{-}= 2\,(ia|kp)\,\bar t_{ia,kl},\\
c_1^{y}[j,b] &\mathrel{+}= 2\,f^{x}_{ia}\,\bar t_{ia,jb}, &
c_0 &\mathrel{-}= (ia|jb)\,\bar t_{ia,jb}.
\end{aligned}
$$

Second order, $\tfrac12[[\hat F,\hat A],\hat A]_1$, with the intermediates
$V_{ia,jd} = f^{x}_{ac}\,\bar t_{ic,jd}$ and $O_{ia,jd} = f^{x}_{ik}\,\bar t_{ka,jd}$:

$$
\begin{aligned}
c_1^{y}[j,k] &\mathrel{+}= t_{ia,jb}\,(V-O)_{ia,kb}, &
c_1^{x}[i,k] &\mathrel{+}= t_{ia,jb}\,V_{ka,jb},\\
c_1^{x}[a,c] &\mathrel{+}= t_{ia,jb}\,O_{ic,jb}, &
c_1^{y}[b,d] &\mathrel{+}= t_{ia,jb}\,(O-V)_{ia,jd},\\
c_0 &\mathrel{+}= t_{ia,jb}\,(O-V)_{ia,jb}, &
c_1^{x}[i,a] &\mathrel{+}= \bar t_{ia,jb}\,y^{y}_{jb},\quad
y^{y}_{jb}=\textstyle\sum_{x'} f^{x'}_{kc}\,\bar t^{\,x'y}_{kc,jb},
\end{aligned}
$$

and, with $R^{\rm occ}_{ik} = t_{ia,jb}\bar t_{ka,jb}$ and
$R^{\rm vir}_{ac} = t_{ia,jb}\bar t_{ic,jb}$ accumulated over the blocks with
first spin $x$,

$$
c_1^{x}[p,i] \mathrel{-}= f^{x}_{pk}R^{\rm occ}_{ik},\qquad
c_1^{x}[p,c] \mathrel{-}= f^{x}_{pa}R^{\rm vir}_{ac}.
$$

Finally $v = \tfrac12(c_1 + c_1^{T})$. The second-order part needs no
two-electron integrals.

`bch_order=1` keeps only the first commutator with half weight,
$\hat H + \tfrac12[\hat H,\hat A]_1$. It was the default of some modules of
version 0.2; the second-order form is the default now.

### Cost

One cycle needs the integrals $(ia|jb)$, $(ia|pb)$ and $(ia|jp)$ and
contractions that scale as $o^2v^2N$ (first order) and $o^2v^3$ (second
order), i.e. $O(N^5)$ like MP2. With exact integrals the first pair is
transformed once per cycle, $(ia|\mu\nu)$; with density fitting the
three-index tensors $B^Q_{pa}$, $B^Q_{ip}$ are used. Memory is dominated by
the amplitudes, $O(o^2v^2)$, and the three-index or half-transformed
integrals.

## 2. O2BMP2

Same-spin amplitudes are dropped and opposite-spin amplitudes are scaled,
$c_{\rm SS}=0$, $c_{\rm OS}=c$. The scaling acts on the amplitudes that enter
$v$, so it changes orbitals and orbital energies and not only the energy.
There is no universal $c_{\rm OS}$; the published choices are 1.2
(singlet–triplet gaps, charge-transfer excitations), 1.1 (ionization
potentials) and 1.7 (inverted singlet–triplet gaps). `pycmf.O2BMP2` therefore
has no default. A Laplace-transform $O(N^4)$ algorithm is not implemented.

## 3. OBDH and O2BDH

For a double-hybrid functional with exact-exchange fraction $a_x$ and
correlation weight $a_c$,

$$
\bar f = f^{\rm KS}[a_x\,{\rm HF} + (1-a_x)X,\ (1-a_c)C] + a_c\,v_{\rm OBMP2},
\qquad E = E^{\rm KS} + a_c\,\big(c_0 + \textstyle\sum_i v_{ii}\big).
$$

$v_{\rm OBMP2}$ is built with the HF Fock matrix of the current orbitals and
with amplitudes whose denominators are the eigenvalues of $\bar f$. With
$a_x=a_c=1$ the method is OBMP2; with $a_c=0$ it is the hybrid functional.
O2BDH uses opposite-spin amplitudes only, scaled by $c_{\rm OS}$.

The Kohn–Sham part is evaluated by PySCF. The default is B2PLYP; the global
double hybrids below are available by name (`xc='B2GP-PLYP'`), any other
combination through the four parameters or a functional string:

| Name | $a_x$ | $a_c$ | exchange | correlation | Reference |
| --- | --- | --- | --- | --- | --- |
| B2PLYP | 0.53 | 0.27 | B88 | LYP | Grimme, *J. Chem. Phys.* **124**, 034108 (2006) |
| mPW2PLYP | 0.55 | 0.25 | mPW91 | LYP | Schwabe and Grimme, *Phys. Chem. Chem. Phys.* **8**, 4398 (2006) |
| B2T-PLYP | 0.60 | 0.31 | B88 | LYP | Tarnopolsky et al., *J. Phys. Chem. A* **112**, 3 (2008) |
| B2GP-PLYP | 0.65 | 0.36 | B88 | LYP | Karton et al., *J. Phys. Chem. A* **112**, 12868 (2008) |
| B2K-PLYP | 0.72 | 0.42 | B88 | LYP | Tarnopolsky et al., *J. Phys. Chem. A* **112**, 3 (2008) |
| PBE0-DH | 0.5 | 0.125 | PBE | PBE | Brémond and Adamo, *J. Chem. Phys.* **135**, 024106 (2011) |
| PBE-QIDH | $3^{-1/3}$ | 1/3 | PBE | PBE | Brémond et al., *J. Chem. Phys.* **141**, 031101 (2014) |
| PBE0-2 | $2^{-1/3}$ | 0.5 | PBE | PBE | Chai and Mao, *Chem. Phys. Lett.* **538**, 121 (2012) |

These parameters were fitted with a perturbative second-order energy on
Kohn–Sham orbitals; they have not been re-optimised for the self-consistent
scheme.

### Embedding in DFT

In the projection-based embedding (`pycmf.embed`, experimental) the two
levels have their own functionals. A Kohn–Sham calculation of the full system
with the low-level functional `xc_env` gives the densities $\gamma_A$,
$\gamma_B$ of the subsystem and the environment (SPADE partition) and the
embedding potential

$$
v_{\rm emb} = v^{\rm env}[\gamma_A+\gamma_B]-v^{\rm env}[\gamma_A]+\mu\,S\gamma_BS .
$$

The subsystem is then treated with OBMP2 or OBDH (its own double hybrid) in
this potential, optionally in a virtual space truncated by concentric
localization, and

$$
E = E^{\rm high}[\gamma_A'] + E^{\rm env}[\gamma_A'+\gamma_B]-E^{\rm env}[\gamma_A']
 + \mu\,{\rm tr}(\gamma_A'S\gamma_BS).
$$

Two limits are exact and tested: with all atoms in the subsystem $E$ is the
energy of the high-level method, and with the same functional on both levels
and $a_c=0$ it is the Kohn–Sham energy of the full system.

OBDH and O2BDH have not been described in a publication yet; this page and the
module docstring are their definition.

## 4. State-specific excited states

The correlated Fock operator is a one-body operator, so non-Aufbau
determinants can be optimised as in ΔSCF. The calculation starts from a
MOM-HF determinant; after each diagonalisation the orbitals with the largest
overlap with the occupied space of the previous cycle (`'mom'`) or of the
initial determinant (`'imom'`) are occupied. Excitation energies are
differences of total energies.

## 5. One-body downfolding (OBDF)

Let $T_{\rm act}$ be the amplitudes with all four indices in an active space.
The external correlation potential and constant are

$$
v^{\rm ext} = v[T] - v[T_{\rm act}],\qquad c_0^{\rm ext} = c_0[T]-c_0[T_{\rm act}],
$$

and the active-space Hamiltonian is

$$
\hat H_{\rm act} = E_{\rm core} + \sum_{tu}\bar h_{tu}\,\hat a^\dagger_t\hat a_u
 + \tfrac12\sum_{tuvw}(tu|vw)\,\hat a^\dagger_t\hat a^\dagger_v\hat a_w\hat a_u,
$$

$$
\bar h = h + v^{\rm HF}_{\rm core} + v^{\rm ext},\qquad
E_{\rm core} = E_{\rm nuc} + E^{\rm HF}_{\rm core} + c_0^{\rm ext}
 + \sum_{i\in{\rm core}} v^{\rm ext}_{ii}.
$$

Only the one-body integrals are dressed. Two limits hold exactly and are
tested: if the active space contains all orbitals, $\hat H_{\rm act}=\hat H$
(FCI energy); if it is empty, $E_{\rm core}=E_{\rm OBMP2}$. For any active
space the reference determinant has the energy
$\langle\Phi|\hat H_{\rm act}|\Phi\rangle = E_{\rm OBMP2}-E_c[T_{\rm act}]$,
with $E_c[T_{\rm act}]=c_0[T_{\rm act}]+\sum_i v[T_{\rm act}]_{ii}$: the
correlation energy of the active double excitations is what is left to the
active-space solver. The k-point version is described in
[kpoints.md](kpoints.md).

## 6. Properties

- *HF-like density*: the density of the OBMP2 determinant.
- *MP2-like density*: $\gamma_{\rm oo}=1-R^{\rm occ}$,
  $\gamma_{\rm vv}=R^{\rm vir}$ with the converged amplitudes (unrelaxed).
  After one cycle it equals the unrelaxed MP2 density.
- $\langle S^2\rangle$ of the determinant, optionally with the first-order
  doubles correction.
- Dipole moments and isotropic hyperfine couplings as expectation values over
  either density.
- Koopmans-type ionization potentials and electron affinities from the
  eigenvalues of $\bar f$.

## References

1. T. N. Lan and T. Yanai, *J. Chem. Phys.* **138**, 224108 (2013).
2. L. N. Tran, *J. Phys. Chem. A* **125**, 9242 (2021).
3. L. N. Tran, *Phys. Chem. Chem. Phys.* **24**, 19393 (2022).
4. N. T. Le and L. N. Tran, *J. Phys. Chem. A* **127**, 5222 (2023).
5. N. T. Tran, H. T. Nguyen and L. N. Tran, *J. Phys. Chem. A* **128**, 1543 (2024).
6. N. T. Tran and L. N. Tran, *J. Chem. Phys.* **162**, 104109 (2025).
7. S. Grimme, *J. Chem. Phys.* **124**, 034108 (2006).
8. A. T. B. Gilbert, N. A. Besley and P. M. W. Gill, *J. Phys. Chem. A* **112**, 13164 (2008).
9. G. M. J. Barca, A. T. B. Gilbert and P. M. W. Gill, *J. Chem. Theory Comput.* **14**, 1501 (2018).
