#!/usr/bin/env python
'''
Compute selected quadratic-response transition dipoles and plot a RIXS map.

The first positional argument is the combined SCF/QR checkpoint. It defaults
to the file written by tdrks_2p3d.py.
'''

import sys

import numpy as np
import pyscf.rixs
import pyscf.zora
from pyscf import dft, scf
from pyscf.data.nist import HARTREE2EV
from pyscf.rixs import (
    select_significant_peaks,
    select_states,
    rixs_amplitudes,
    rixs_map,
)

# State-selection windows.  Only states inside these windows are included in
# the response calculation.  These should be broad enough for all maps that
# will be plotted from this calculation.
INTERMEDIATE_STATE_WINDOW = (750, 800)  # eV
FINAL_STATE_WINDOW = (1, 5)  # eV

# Plotting windows.  These can be changed without changing which states were
# selected for the response calculation above.
INCIDENT = (775, 783)  # eV
TRANSFER = (2, 6)  # eV

# Any number of slices is allowed.
T_SLICE = (3.8, 4.4)
I_SLICE = (779.5,)
NPOINTS = 400

PRECOMPUTE_GXC = False
NAME = 'tdrks_2p3d'
QR_CHECKPOINT = NAME + '_qr.chk'
FILENAME = NAME + '_rixs_map'
TITLE = 'Co³⁻'

args = sys.argv[1:]
NO_PLOT = '--no-plot' in args
args = [arg for arg in args if arg != '--no-plot']
if len(args) > 1:
    raise SystemExit(
        'Usage: quadratic_rixs_map.py [qr_checkpoint] '
        '[--no-plot]'
    )


def load_reference(chkfile):
    '''Restore the combined SCF/QR/dipole checkpoint.'''
    mol = scf.chkfile.load_mol(chkfile)
    template = dft.RKS(mol, xc='PBE0').set(max_cycle=200)
    template.grids.level = 4
    return pyscf.rixs.RIXS.from_chk(
        chkfile,
        template,
        precompute_gxc=PRECOMPUTE_GXC,
    )



qr_file = args[0] if len(args) > 0 else QR_CHECKPOINT

rixs = load_reference(qr_file)
mf = rixs.mf
qr = rixs.qr
qr.approximation = 'Nascimento'

e_n_all = np.asarray(qr.manifold_n.e)
e_f_all = np.asarray(qr.manifold_m.e)
n_states = select_states(
    e_n_all,
    INTERMEDIATE_STATE_WINDOW,
    window_unit='eV',
)
f_states = select_states(
    e_f_all,
    FINAL_STATE_WINDOW,
    window_unit='eV',
)
if len(n_states) == 0 or len(f_states) == 0:
    raise RuntimeError(
        f'Energy windows selected no states: '
        f'n={len(n_states)} from {INTERMEDIATE_STATE_WINDOW} eV, '
        f'f={len(f_states)} from {FINAL_STATE_WINDOW} eV'
    )
print(
    f'Selected {len(n_states)} intermediate and {len(f_states)} final states',
    flush=True,
)

n_mu_0 = rixs.ground_transition_dipoles(n_states)
f_mu_n = rixs.transition_dipoles(n_states, f_states)

f_fn = rixs_amplitudes(f_mu_n, n_mu_0)
max_f_fn = np.amax(np.abs(f_fn))
if max_f_fn == 0:
    raise RuntimeError('All selected RIXS amplitudes are zero')
sig_peaks = select_significant_peaks(f_fn)
print(
    f'Computed |Ffn|, max: {max_f_fn * 1e9:.5f}e-9; '
    f'significant pairs: {len(sig_peaks):g}',
    flush=True,
)

with open(FILENAME + '.csv', 'w') as file:
    file.write('f,n,omega_n_eV,omega_f_eV,Ffn(θ=0),'
               'fμn_x,fμn_y,fμn_z\n')
    for f_pos, n_pos in sig_peaks:
        mu = f_mu_n[:, f_pos, n_pos]
        file.write(
            f'{f_states[f_pos]:g},{n_states[n_pos]:g},'
            f'{e_n_all[n_states[n_pos]] * HARTREE2EV:.9f},'
            f'{e_f_all[f_states[f_pos]] * HARTREE2EV:.9f},'
            f'{f_fn[f_pos, n_pos]:.14f},'
            f'{mu[0]:12.9f},{mu[1]:12.9f},{mu[2]:12.9f}\n'
        )

if NO_PLOT:
    print('Skipping plots (--no-plot)', flush=True)
    raise SystemExit(0)

try:
    import matplotlib
    matplotlib.use('svg')
    from matplotlib import cm
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif'})
except ModuleNotFoundError as err:
    raise SystemExit(
        'Plotting requires matplotlib; use --no-plot '
        'to keep the QR/CSV calculation only'
    ) from err


def _layout_figure(fig):
    fig.subplots_adjust(
        left=0.16,
        right=0.90,
        bottom=0.14,
        top=0.90,
    )


def plot_map(X, Y, Z):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    cmap = cm.summer
    norm = cm.colors.Normalize(vmin=abs(Z).min(), vmax=abs(Z).max())
    levels = np.linspace(abs(Z).min(), np.amax(Z), 22)
    cset1 = ax.contourf(
        X,
        Y,
        Z,
        levels,
        norm=norm,
        cmap=cmap.resampled(len(levels) - 1),
    )
    ax.contour(X, Y, Z, cset1.levels, colors='k', linewidths=.4)
    ax.set_xlim(INCIDENT)
    ax.set_ylim(TRANSFER)
    ax.set_ylabel('Energy Transfer (eV)')
    ax.set_xlabel('Incident energy (eV)')
    ax.tick_params(
        axis='both',
        which='both',
        direction='in',
        top=True,
        right=True,
    )
    ax.set_title(TITLE, y=1.0, x=0.16, pad=-14)
    ax.vlines(
        x=I_SLICE,
        ymin=TRANSFER[0],
        ymax=TRANSFER[1],
        colors='white',
        linestyles=(0, (10, 10)),
        linewidth=.3,
    )
    ax.hlines(
        y=T_SLICE,
        xmin=INCIDENT[0],
        xmax=INCIDENT[1],
        colors='white',
        linestyles=(0, (10, 10)),
        linewidth=.3,
    )
    _layout_figure(fig)
    plt.savefig(FILENAME + '.svg')


_SLICE_CURVE_STYLES = (
    {'color': '#000000', 'linestyle': '-'},
    {'color': '#000000', 'linestyle': '--'},
    {'color': '#555555', 'linestyle': '-.'},
    {'color': '#555555', 'linestyle': ':'},
)


def plot_slices(X, Y, Z):
    if len(I_SLICE) == 0 or len(T_SLICE) == 0:
        raise ValueError("Must provide slices")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    slice_idxs = np.asarray([
        np.argmin(np.abs(X[0] - I_SLICE[i])) for i in range(len(I_SLICE))
    ])
    max_abs = np.amax(Z[:, slice_idxs])
    for i in range(len(I_SLICE)):
        ax1.plot(
            Y[:, 0],
            Z[:, slice_idxs[i]] / max_abs,
            label=f"ω′={I_SLICE[i]:.1f} eV",
            **_SLICE_CURVE_STYLES[i % len(_SLICE_CURVE_STYLES)],
        )
    ax1.set_ylim((0, 1.04))
    ax1.margins(0)
    ax1.grid(visible=True, linewidth=.4, linestyle=':', color='#999999')
    ax1.set_ylabel('Intensity (normalized)')
    ax1.set_xlabel('Energy transfer (eV)')
    ax1.tick_params(
        axis='both',
        which='both',
        direction='in',
        top=True,
        right=True,
    )
    ax1.legend(title=TITLE, loc='upper left')

    slice_idxs = np.asarray([
        np.argmin(np.abs(Y[:, 0] - T_SLICE[i])) for i in range(len(T_SLICE))
    ])
    max_abs = np.amax(Z[slice_idxs])
    for i in range(len(T_SLICE)):
        ax2.plot(
            X[0],
            Z[slice_idxs[i]] / max_abs,
            label=f'ω={T_SLICE[i]:.1f} eV',
            **_SLICE_CURVE_STYLES[i % len(_SLICE_CURVE_STYLES)],
        )
    ax2.set_ylim((0, 1.04))
    ax2.set_yticklabels([])
    ax2.set_xlabel('Incident energy (eV)')
    ax2.tick_params(
        axis='both',
        which='both',
        direction='in',
        top=True,
        right=True,
    )
    ax2.grid(visible=True, linewidth=.4, linestyle=':', color='#999999')
    ax2.legend(loc='upper left')
    ax2.margins(0)

    fig.tight_layout(pad=1.5, w_pad=2.0)
    plt.savefig(FILENAME + '_slice.svg')


print('Computing RIXS map', flush=True)
gamma = 1
X, Y, Z = rixs_map(
    f_fn,
    e_n_all[n_states] * HARTREE2EV,
    e_f_all[f_states] * HARTREE2EV,
    np.linspace(INCIDENT[0], INCIDENT[1], NPOINTS),
    np.linspace(TRANSFER[0], TRANSFER[1], NPOINTS),
    peaks=sig_peaks,
    incident_broadening=gamma,
    transfer_broadening=1,
)

plot_map(X, Y, Z)
plot_slices(X, Y, Z)
