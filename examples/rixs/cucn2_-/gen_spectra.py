#!/usr/bin/env python

'''
File that creates an SVG for csv files containing energy of transition
(w) in first column and oscillator strengths (f) in second column.
'''

TITLE = '[Cu(CN)₂]⁻ XAS ZORA (no SO)'

import sys
import numpy as np
np.set_printoptions(suppress=True, precision=7, linewidth=170)
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
plt.rcParams.update({ "font.family": "serif" })

# Homogeneous broadening parameter (HWHM for distribution)
BROADENING = 1
NUM_POINTS = 5000

def cauchy(x, γ):
    '''
    PDF of Cauchy distribution: location (x), scale(γ)
    Used to describe inhomogeneous broadening.
    '''
    return 1/(np.pi*(γ + x**2 / γ))

csv_files = []
if len(sys.argv) < 2:
    # TODO: get all csv files in cwd
    pass
else:
    for i in range(1, len(sys.argv)):
        csv_files.append(sys.argv[i])

assert len(csv_files) != 0

wf_dict = {}
for f in csv_files:
    clean = f.split('.')[0]
    try:
        wf_dict[clean] = np.loadtxt(f, delimiter=',')
    except Exception as err:
        print(err)
        print('Could not import csv file:', f+'.csv')

for filename, wf in wf_dict.items():
    w, f = wf.T
    w = w[f > 1e-4]
    f = f[f > 1e-4]
    if f.shape[0] < 2:
        print('ERROR: no non-zero excitations to plot for', filename+'.csv')
        continue

    # turn atomic units to eV
    w *= 27.21138602 # NIST value
    min_e = np.amin(w) - 3
    max_e = np.amax(w) + 3
    window = np.where(w < max_e)[0]
    w = w[window]
    f = f[window]

    X = np.linspace(min_e, max_e, NUM_POINTS)
    Y = np.zeros(X.shape)

    if not (0.01 < BROADENING / (max_e - min_e) < .5):
        print("WARNING: broadening setting is whack")
        print(f"  FWHM = {2 * BROADENING:10.8f}   XLIM = ({min_e}, {max_e})")

    # Turn sticks into plot
    for i, x in enumerate(X):
        for wi, fi in zip(w, f):
            Y[i] += cauchy(x-wi, BROADENING) * fi

    plt.figure(figsize=(8,5), dpi=80)
    graph = plt.subplot()
    graph.plot(X, Y, linewidth=2, color='#000000')
    markerline, stemline, baseline = graph.stem(w, f, basefmt=" ")
    markerline.set_color('#777777')
    markerline.set_markersize(1.4)
    stemline.set_color('#777777')
    stemline.set_linewidth(.8)
    graph.set_title(TITLE)
    graph.xlim = (min_e, max_e)
    # graph.set_ylim(bottom=0,top=.010)
    # graph.set_yticks(np.arange(0,.011,.002))
    # graph.set_xticks(np.arange(min_e,max_e+1,5))
    # graph.set_xticks(np.arange(min_e+1,max_e,2))
    graph.grid(visible=True, linestyle=(0,(5,5)), linewidth=.5, color='#AAAAAA')
    graph.set_xmargin(0)
    graph.set_ymargin(0)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intensity')
    plt.tight_layout()
    plt.savefig(filename + '.svg')

