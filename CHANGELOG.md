# Changelog

## v0.5 - 2026-??-??

### Added

**ZORA:**
- Added ZORA gradients including for spin_orbit.
- Support for ZORA geometry optimizations. Requires PySCF>2.14 for GHF/GKS
  gradients/optimizations.
- MO energy scaling correction.
- `ZORA_SCF.undo_zora()` so ZORA mixin structure follows PySCF conventions
  for other mixins.

**CVS:**
- `core_window` option for CVS: specify orbitals by energy range.

**RIXS:**
- Added `pyscf.rixs` module and `RIXS` class.
- checkpoint save/load. Requires QR and mf object. Compatible with `ZORA_SCF` objects.
- `rixs_amplitude` is the amplitude for terms in the Kramers-Heisenberg equation.
- `rixs_map` computes the full map mesh from KH equation.
- Computes ground-to-intermediate and intermediate-to-final states for you.

**Other:**
- Code coverage on CI and coverage badge in README.
- PySCF features explicitly checked. Different PySCF versions support
  different capabilities.

### Fixed

- CVS class no longer modifies your orbitals. It uses a mixin structure.
- GGA Eager Gxc memory estimate allocated way more memory than necessary.
- No warning appeared when precompute_gxc and self.G is None.
- `test` dirs were importable modules for editable installs.
- precomputing Gxc was not idempotent.

### Changed

- `.zora()` returns `mf`, so you can now do `mf = scf.RHF(mol).zora()`.
- ZORA options now live on `mf.with_zora`.
- Tests moved from `pyscf/*/test` to `tests/*`.
- CVS class now uses PySCF's builtin `frozen` attribute. Minimum PySCF version
  supported for CVS is 2.7.
- Default `n_states` for CVS is now (nocc*nvirt) when using `direct_diag`.
- Use Y=0 convention for TDA, following PySCF conventions.
