#!/usr/bin/env python
'''
Plot results from LiH-all_approx.py
'''

from pathlib import Path

import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
import numpy as np

datafile = Path(__file__).with_name('LiH-all_approx.npz')
data = np.load(datafile)

bond_lengths = data['bond_lengths']
w01 = data['w01']
w14 = data['w14']
fci_w01 = data['fci_w01']
fci_w14 = data['fci_w14']
fci_tdip14 = data['fci_tdip14']
tdip14 = data['tdip14']
pw_tdip14 = data['pw_tdip14']
g0_tdip14 = data['g0_tdip14']

plt.rcParams.update({ "font.family": "serif" })
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6,8), dpi=85, gridspec_kw={'height_ratios':[.6,1]})
ax1.set_title('Approximate $\\mu_{14}$ from TDDFT')
ax1.plot(bond_lengths, w01, label='$\\Omega_{01}^\\text{TDDFT}$', color='tab:blue', linewidth=2)
ax1.plot(bond_lengths, w14, label='$\\Omega_{14}^\\text{TDDFT}$', color='tab:blue', linestyle='--', linewidth=2)
ax1.plot(bond_lengths, fci_w01, label='$\\Omega_{01}^\\text{FCI}$', color='tab:red', linewidth=2)
ax1.plot(bond_lengths, fci_w14, label='$\\Omega_{14}^\\text{FCI}$', color='tab:red', linestyle='--', linewidth=2)
ax1.set_ylabel('E (eV)')
ax1.set_yticks([1.5,2,2.5,3,3.5])
ax1.set_ylim((1.5,3.5))
ax1.set_xticks(np.arange(2,3.1,.2))
ax1.grid(visible=True, linestyle=':', linewidth=.8)
ax1.set_xticklabels([])
ax1.margins(0)
ax1.legend()

ax2.plot(bond_lengths, fci_tdip14, label='FCI', color='tab:red', linewidth=2)
ax2.plot(bond_lengths, tdip14, label='Full QR', color='tab:blue', linewidth=2)
ax2.plot(bond_lengths, pw_tdip14, label='pseudo-wav.', color='tab:green', linewidth=2)
ax2.plot(bond_lengths, g0_tdip14, label='$\\vert X^{(\\alpha\\beta)},Y^{(\\alpha\\beta)}\\rangle=0$', color='tab:orange', linewidth=2)
ax2.set_ylabel('$\\mu_{14}$ (a.u.)')
ax2.set_xlabel('Bond length (Å)')
ax2.set_xticks(np.arange(2,3.1,.2))
ax2.margins(0)
ax2.set_yticks(np.arange(0,.7,.1))
ax2.set_ylim((0,.7))
ax2.grid(visible=True, linestyle=':', linewidth=.8)
ax2.legend()

fig.tight_layout()
outfile = Path(__file__).with_name('LiH-all_approx.svg')
plt.savefig(outfile)
print(f'Saved figure to {outfile}')
