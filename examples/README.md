# Examples

| Script | Shows |
| --- | --- |
| `01-obmp2.py` | restricted and unrestricted OBMP2, exact and density-fitted integrals |
| `02-o2bmp2.py` | O2BMP2 and the role of `c_os`: singlet–triplet gap of CH₂ |
| `03-excited_state.py` | state-specific excited state with MOM |
| `04-obdf.py` | one-body downfolding along the N₂ dissociation curve, FCIDUMP export |
| `05-obdh.py` | OBDH and O2BDH, double hybrids by name or by parameters, open shells |
| `06-properties.py` | densities, dipole moment, ⟨S²⟩ and hyperfine couplings of CN |
| `experimental/` | RO-OBMP2, projection embedding, k-point OBMP2 and downfolding (under development) |

`reference_output/` holds the expected output of every script.
`python check_examples.py` runs the stable scripts and compares all printed
numbers with the reference (`--all` adds the experimental ones).
