# Dicyanocuprate RIXS example

This directory demonstrates an end-to-end quadratic-response RIXS workflow
using $\ce{[Cu(CN)2]-}$ in `cucn2_-`. Perturbative spin-orbit ZORA is
not yet supported, and generalized SCF is not supported for RIXS yet.

The example is intended to show how the plugin connects:

1. ZORA DFT reference
2. CVS-TDA calculations for the 2p and 3d manifolds
3. ground and excited transition dipoles
4. state-pair evaluation
5. peak selection and
6. RIXS map from averaged KH equation

## Files

`cucn2_-/tdrks_2p3d.py`
: Runs the SCF calculation, stability analysis, and CVS-TDA calculations.
  It constructs the QR/RIXS object and saves the combined checkpoint and a
  Molden orbital file. It also writes intermediate-manifold XAS sticks for
  `gen_spectra.py`.

`cucn2_-/quadratic_rixs_map.py`
: Loads the combined checkpoint, selects intermediate and final states using
  eV windows, computes the selected transition dipoles and RIXS pair factors,
  writes a CSV of significant peaks, and optionally creates a map and slice
  plots. Use `--no-plot` to stop after the numerical calculation and CSV
  output.

`cucn2_-/gen_spectra.py`
: Optional plotting utility for CSV files containing excitation energies and
  oscillator strengths. It applies Cauchy broadening and writes an SVG XAS
  spectrum. It is separate from the RIXS map calculation.

`cucn2_-/reference_*.svg`
: These are SVGs showing what the output would look like.

## Dependencies

From the repository root, install the plugin into the environment used for
the calculation:

```bash
python -m pip install -e '.[cvs]'
python -m pip install matplotlib
```

PySCF and NumPy are installed as dependencies of the project. The example
uses the CVS functionality, so use a PySCF version supporting the CVS API
used here; the project’s CVS extra currently requires PySCF 2.10 or newer.

`matplotlib` is needed only for the RIXS map and XAS plots. The `--no-plot`
mode does not require it.

## Running the example

Run the commands from this directory:

```bash
cd examples/rixs/cucn2_-
python tdrks_2p3d.py
```

The first command performs the expensive SCF and CVS-TDA calculations. When
run directly, it writes:

- `tdrks_2p3d_qr.chk`: combined SCF/QR/ZORA/dipole checkpoint;
- `tdrks_2p3d.molden`: Molden orbital file;
- `sticks_tdrks_2p3d.csv`: intermediate excitation energies in Hartree and
  oscillator strengths.

```bash
python quadratic_rixs_map.py
```

This command writes `tdrks_2p3d_rixs_map.csv`. It holds significant `(f, n)`
peaks, energies, and final/intermediate transition-dipole moments. It also
writes the SVGs:

- `tdrks_2p3d_rixs_map.svg`: the two-dimensional RIXS map;
- `tdrks_2p3d_rixs_map_slice.svg`: incident- and transfer-energy slices.

The plotting windows and state-selection windows are configured separately at
the top of `quadratic_rixs_map.py`. The numerical state selection is performed
before the selected QR transition-dipole calculations.

To use the optional XAS plotting utility, provide a CSV with excitation
energies in Hartree in the first column and oscillator strengths in the
second column:

```bash
python gen_spectra.py sticks_tdrks_2p3d.csv
```

This writes `sticks_tdrks_2p3d.svg` alongside the input CSV.
