# Migrating from pyCMF 0.2

| 0.2 | 1.0 |
| --- | --- |
| `OBMP2(mf)`, `OBMP2_slow`, `OBMP2_einsum` | `pycmf.OBMP2(mf)` (RHF reference) |
| `UOBMP2(mf)`, `UOBMP2_slow` | `pycmf.OBMP2(mf)` (UHF/ROHF reference) |
| `DFOBMP2`, `DFUOBMP2`, `DFUOBMP2_einsum` | `pycmf.OBMP2(mf_df)` or `.density_fit()` |
| `UOBMP2_SCS`; `css`, `cos` | `c_ss`, `c_os`; `pycmf.O2BMP2(mf, c_os=...)` |
| `UOBMP2_mom`, `UOBMP2_mom_diis`, `DFUOBMP2_mom*`; `mom_select=True` | `occupation='imom'` |
| `UHF_mom_diis` | `pyscf.scf.addons.mom_occ` |
| `second_order=True` / `False`, `ampf` | `bch_order=2` / `1` |
| `thresh`, `niter`, `shift` | `conv_tol`, `max_cycle`, `level_shift` |
| `mp.ene_tot` | `mp.e_tot` |
| `kernel()` returning a tuple | `kernel()` returns `e_tot`; everything else is an attribute |
| `OBMP2_downfold` / `UOBMP2_downfold` with `mo_coeff`, `fock_hf`, `tmp1`, `tmp1_bar`, `c0_tot`, `c1` copied by hand, `mcscf.sort_mo` | `pycmf.OBDF(mp, ncas, nelecas, cas_list=...)` |
| `h1mo_act_eff`, `h2mo_act`, `ene_inact` | `get_h1eff()` → `(h1eff, e_core)`, `get_h2eff()` |
| `caslist` counted from 1 | `cas_list` counted from 0 |
| `DFTOBMP2`, `OBDH_CL` with `alphaa=(ax, ac)` | `pycmf.OBDH(mf, ax=..., ac=...)` or `pycmf.OBDH(mf, xc='B2PLYP')` |
| `OBDH_CL` / `OBMP2_CL` with `use_embed=True`, `use_cl`, `n_shells`, `xc_env`, `mu` | `pycmf.embed.EmbeddedOBDH` / `EmbeddedOBMP2`; the subsystem now has its own functional (`xc='B2PLYP'`), `xc=xc_env` gives the 0.2 numbers |
| `make_fc(dm)` | `mp.hfcc(correlated=False)` |
| `make_rdm1(use_t2=...)` | `mp.make_rdm1(correlated=...)` |
| `ROOBMP2` (`OBMP/ro_obmp2.py`) | `pycmf.mol.roobmp2.ROOBMP2` |
| `KOBMP2` (k-point, restricted) | `pycmf.pbc.KOBMP2`: `niter`→`max_cycle`, `thresh`→`conv_tol`, `shift`→`level_shift`, `ene_tot`→`e_tot` (the old names still work); `kernel()` returns the energy; density-fitted integrals |
| `KUOBMP2` | `pycmf.pbc.KUOBMP2` (interface unchanged) |
| `KROBDF` / `krobdf.OBMP2` with `nact`, `nocc_act`, `fock_hf`, `c0_tot`, `c1` copied by hand; `h1mo_act_eff`, `h2mo_act`, `ene_inact` | `mp.downfold(ncas, nelecas)` → `pycmf.pbc.KOBDF`: `get_h1eff()` → `(h1eff, e_core)`, `get_h2eff()` (same normalisation, 1/Nk) |
| `make_fock_df(scf_active=True)`, `make_ipea()` | `ham.scf()`, `ham.self_energy()` + `ham.quasiparticle()` |
| `Supercell`, `solve_fci`, `solve_ccsd` | `ham.supercell()`, `ham.kernel()`, `ham.as_scf()` |

Example, 0.2:

```python
mp = UOBMP2_mom_diis(mf)
mp.second_order = True; mp.css = 0.0; mp.cos = 1.2
mp.mom_select = True; mp.thresh = 1e-8
e = mp.kernel()
```

1.0:

```python
mp = pycmf.O2BMP2(mf, c_os=1.2, occupation='imom')
e = mp.kernel()
```
