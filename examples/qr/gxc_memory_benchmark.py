#!/usr/bin/env python
'''
Benchmark Gxc per-block memory: predicted vs actual (tracemalloc).

Two parameter sweeps on H2O (def2-svp, max_memory=6000 MB):

* **Grid level** (x-axis: number of grid points) — fixed manifolds.
* **Occupied orbitals on manifold m** (x-axis: nocc_m) — freeze 0–4 core
  orbitals on td_m only; td_n keeps all occupied orbitals active.

Crossed with LDA/PBE0 and eager/lazy Gxc evaluation → 8 comparison plots.

Caveats
-------
* At high grid levels, blksize may hit the 1200×56 cap and both curves
  flatten vs ngrids.
* Eager occ sweep: predicted scales with nocc_m via the nocc_n·nocc_m·nvir²
  term.
* Lazy occ sweep: the contract_v predictor does not depend on nocc_m; a flat
  predicted line vs varying actual documents that gap.

Usage
-----
Run the full sweep and plot::

    python examples/qr/gxc_memory_benchmark.py --plot

Run parametrized tests (slow)::

    pytest examples/qr/gxc_memory_benchmark.py -m slow
'''

from __future__ import annotations

import argparse
import os
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import pytest
from pyscf import dft, gto
from pyscf.qr import QR
from pyscf.qr.rhf import estimate_gxc_block_peak
from pyscf.tdscf import RPA

H2O_ATOM = 'O 0 0 0; H 0 0 0.96; H 0 0 -0.96'
BASIS = 'def2-tzvp'
MAX_MEMORY_MB = 6000
NSTATES = 4
GRID_LEVELS = (1, 3, 5, 7)
FROZEN_M_VALUES = (0, 1, 2, 3, 4)
XC_VALUES = ('LDA', 'PBE0')
XC_IDS = ('LDA', 'GGA')
EAGER_VALUES = (True, False)
EAGER_IDS = ('eager', 'lazy')

DATAFILE = Path(__file__).with_name('gxc_memory.npz')
FIGFILE = Path(__file__).with_name('gxc_memory.svg')


def _log(msg):
    print(msg, flush=True)


def _make_mf(xc, grid_level):
    _log(f'  SCF start (xc={xc}, grid_level={grid_level})')
    t0 = time.perf_counter()
    mol = gto.M(atom=H2O_ATOM, basis=BASIS, verbose=0, max_memory=MAX_MEMORY_MB)
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = grid_level
    mf.kernel()
    _log(f'  SCF done in {time.perf_counter() - t0:.1f}s '
         f'({mf.grids.size} grid points)')
    return mf


def _measure_gxc_peak(run):
    '''Return peak traced memory (bytes) for a Gxc kernel call.

    Eager precompute resets tracemalloc peak each grid block; capture the
    maximum block peak so the measurement matches the per-block predictor.
    '''
    block_peaks = []
    orig_reset = tracemalloc.reset_peak

    def capture_reset_peak():
        _, peak = tracemalloc.get_traced_memory()
        block_peaks.append(peak)
        orig_reset()

    tracemalloc.start()
    tracemalloc.reset_peak = capture_reset_peak
    try:
        run()
    finally:
        tracemalloc.reset_peak = orig_reset
        _, final_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return max(block_peaks) if block_peaks else final_peak


def run_gxc_memory_case(*, xc, eager, grid_level=5, frozen_m=None):
    '''Run one benchmark case and return predicted/actual peak memory (GB).'''
    mode = 'eager' if eager else 'lazy'
    frozen_label = 'none' if frozen_m is None else frozen_m
    _log(f'  TD/RPA start (frozen_m={frozen_label}, mode={mode})')
    t0 = time.perf_counter()

    mf = _make_mf(xc, grid_level)
    td_n = RPA(mf).run(nstates=NSTATES, verbose=0)

    if frozen_m is None:
        td_m = td_n
    else:
        td_m = RPA(mf, frozen=frozen_m).run(nstates=NSTATES, verbose=0)

    qr = QR(td_n, td_m, precompute_gxc=eager)
    predicted_bytes, info = estimate_gxc_block_peak(mf, qr, eager=eager)
    nocc_m = len(qr.manifold_m.occ_idx)
    _log(f'  TD/RPA done in {time.perf_counter() - t0:.1f}s '
         f'(nocc_m={nocc_m}, blksize={info["blksize"]})')

    _log(f'  Gxc kernel start ({mode})')
    t1 = time.perf_counter()
    if eager:
        actual_bytes = _measure_gxc_peak(lambda: qr.kernel())
    else:
        tracemalloc.start()
        qr.get_2tdm(0, min(1, NSTATES - 1))
        _, actual_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    _log(f'  Gxc kernel done in {time.perf_counter() - t1:.1f}s')

    result = dict(
        xc=xc,
        eager=eager,
        grid_level=grid_level,
        frozen_m=-1 if frozen_m is None else frozen_m,
        ngrids=info['ngrids'],
        nocc_m=nocc_m,
        blksize=info['blksize'],
        mem_per_point=info['mem_per_point'],
        max_memory_mb=info['max_memory_mb'],
        xctype=info['xctype'],
        predicted_gb=predicted_bytes / 1024**3,
        actual_gb=actual_bytes / 1024**3,
    )
    ratio = result['actual_gb'] / result['predicted_gb']
    _log(f'  result: predicted={result["predicted_gb"]:.3f} GB, '
         f'actual={result["actual_gb"]:.3f} GB, ratio={ratio:.2f}')
    return result


def _assert_sane(result):
    ratio = result['actual_gb'] / result['predicted_gb']
    assert result['predicted_gb'] > 0
    assert result['actual_gb'] > 0
    assert 0.1 < ratio < 3.0, (
        f"actual/predicted={ratio:.2f} out of range for {result!r}")


def _maybe_collect(result, sweep):
    if os.environ.get('GXC_MEM_COLLECT') == '1':
        _COLLECTOR.append((sweep, result))


_COLLECTOR: list[tuple[str, dict]] = []


@pytest.mark.slow
@pytest.mark.parametrize('grid_level', GRID_LEVELS)
@pytest.mark.parametrize('xc', XC_VALUES, ids=XC_IDS)
@pytest.mark.parametrize('eager', EAGER_VALUES, ids=EAGER_IDS)
def test_gxc_memory_vs_grid_level(grid_level, xc, eager):
    result = run_gxc_memory_case(xc=xc, eager=eager, grid_level=grid_level)
    _assert_sane(result)
    _maybe_collect(result, 'grid')


@pytest.mark.slow
@pytest.mark.parametrize('frozen_m', FROZEN_M_VALUES)
@pytest.mark.parametrize('xc', XC_VALUES, ids=XC_IDS)
@pytest.mark.parametrize('eager', EAGER_VALUES, ids=EAGER_IDS)
def test_gxc_memory_vs_nocc_m(frozen_m, xc, eager):
    result = run_gxc_memory_case(
        xc=xc, eager=eager, grid_level=5, frozen_m=frozen_m)
    _assert_sane(result)
    _maybe_collect(result, 'occ')


def collect_grid_sweep():
    cases = [
        (grid_level, xc, eager)
        for grid_level in GRID_LEVELS
        for xc in XC_VALUES
        for eager in EAGER_VALUES
    ]
    total = len(cases)
    _log(f'Grid sweep: {total} cases')
    results = []
    t0 = time.perf_counter()
    for i, (grid_level, xc, eager) in enumerate(cases, 1):
        mode = 'eager' if eager else 'lazy'
        _log(f'[grid {i}/{total}] level={grid_level} xc={xc} {mode}')
        results.append(run_gxc_memory_case(
            xc=xc, eager=eager, grid_level=grid_level))
        _log(f'[grid {i}/{total}] finished in '
             f'{time.perf_counter() - t0:.1f}s elapsed')
    _log(f'Grid sweep complete ({time.perf_counter() - t0:.1f}s total)')
    return results


def collect_occ_sweep():
    cases = [
        (frozen_m, xc, eager)
        for frozen_m in FROZEN_M_VALUES
        for xc in XC_VALUES
        for eager in EAGER_VALUES
    ]
    total = len(cases)
    _log(f'Occ sweep: {total} cases')
    results = []
    t0 = time.perf_counter()
    for i, (frozen_m, xc, eager) in enumerate(cases, 1):
        mode = 'eager' if eager else 'lazy'
        _log(f'[occ {i}/{total}] frozen_m={frozen_m} xc={xc} {mode}')
        results.append(run_gxc_memory_case(
            xc=xc, eager=eager, grid_level=5, frozen_m=frozen_m))
        _log(f'[occ {i}/{total}] finished in '
             f'{time.perf_counter() - t0:.1f}s elapsed')
    _log(f'Occ sweep complete ({time.perf_counter() - t0:.1f}s total)')
    return results


def _pack_sweep(results):
    return {key: numpy.array([r[key] for r in results]) for key in results[0]}


def save_results(grid_results, occ_results, path=DATAFILE):
    grid = _pack_sweep(grid_results)
    occ = _pack_sweep(occ_results)
    numpy.savez(path, **{f'grid_{k}': v for k, v in grid.items()},
                **{f'occ_{k}': v for k, v in occ.items()})


def _series_mask(results, xc, eager):
    mask = numpy.ones(len(results), dtype=bool)
    for i, r in enumerate(results):
        if r['xc'] != xc or r['eager'] != eager:
            mask[i] = False
    return mask


def _xc_label(xc):
    return 'GGA' if xc == 'PBE0' else 'LDA'


def _mode_label(eager):
    return 'eager' if eager else 'lazy'


def plot_results(grid_results, occ_results, path=FIGFILE):
    plt.rcParams.update({'font.family': 'serif'})
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=100, sharey='row')

    col_specs = [(xc, eager) for xc in XC_VALUES for eager in EAGER_VALUES]

    for col, (xc, eager) in enumerate(col_specs):
        ax_grid = axes[0, col]
        mask = _series_mask(grid_results, xc, eager)
        subset = [grid_results[i] for i in numpy.where(mask)[0]]
        subset.sort(key=lambda r: r['ngrids'])
        x = [r['ngrids'] for r in subset]
        pred = [r['predicted_gb'] for r in subset]
        actual = [r['actual_gb'] for r in subset]
        ax_grid.plot(x, pred, '--', label='predicted', color='tab:orange')
        ax_grid.plot(x, actual, 'o-', label='actual', color='tab:blue')
        ax_grid.set_title(f'Grid · {_xc_label(xc)} · {_mode_label(eager)}')
        ax_grid.set_xlabel('grid points')
        if col == 0:
            ax_grid.set_ylabel('peak memory (GB)')
        ax_grid.grid(True, linestyle=':', linewidth=0.8)
        ax_grid.legend(fontsize=8)

        ax_occ = axes[1, col]
        mask = _series_mask(occ_results, xc, eager)
        subset = [occ_results[i] for i in numpy.where(mask)[0]]
        subset.sort(key=lambda r: r['nocc_m'])
        x = [r['nocc_m'] for r in subset]
        pred = [r['predicted_gb'] for r in subset]
        actual = [r['actual_gb'] for r in subset]
        ax_occ.plot(x, pred, '--', label='predicted', color='tab:orange')
        ax_occ.plot(x, actual, 'o-', label='actual', color='tab:blue')
        ax_occ.set_title(f'Occ m · {_xc_label(xc)} · {_mode_label(eager)}')
        ax_occ.set_xlabel('nocc_m')
        if col == 0:
            ax_occ.set_ylabel('peak memory (GB)')
        ax_occ.grid(True, linestyle=':', linewidth=0.8)
        ax_occ.legend(fontsize=8)

    fig.suptitle('Gxc per-block memory: predicted vs actual (H2O, def2-svp)')
    fig.tight_layout()
    fig.savefig(path)
    _log(f'Saved figure to {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--plot', action='store_true',
        help='Run both sweeps, save npz, and write the comparison figure.')
    args = parser.parse_args()

    if not args.plot:
        parser.print_help()
        return

    t0 = time.perf_counter()
    _log('Starting Gxc memory benchmark')
    grid_results = collect_grid_sweep()
    occ_results = collect_occ_sweep()
    _log('Saving results')
    save_results(grid_results, occ_results)
    _log(f'Saved data to {DATAFILE}')
    _log('Building plot')
    plot_results(grid_results, occ_results)
    _log(f'Benchmark complete ({time.perf_counter() - t0:.1f}s total)')


if __name__ == '__main__':
    main()
