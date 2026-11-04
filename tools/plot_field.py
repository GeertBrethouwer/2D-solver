#!/usr/bin/env python3
"""
Plot 2D fields written by solver2d (iout_field > 0): data/fld_NNNNNNN_<name>.bin on the dealiased
mx x ny grid (Fortran order (x,y), float64), with data/grid_x.bin and data/grid_y.bin.

    python3 tools/plot_field.py data 8000 u v s1 s2 [--fluct] [-o fields.png]

--fluct subtracts the x-average from each field. Needs numpy and matplotlib.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('data_dir')
ap.add_argument('step', type=int)
ap.add_argument('names', nargs='+')
ap.add_argument('--fluct', action='store_true')
ap.add_argument('-o', '--out', default='fields.png')
a = ap.parse_args()

x = np.fromfile(os.path.join(a.data_dir, 'grid_x.bin'))
y = np.fromfile(os.path.join(a.data_dir, 'grid_y.bin'))
mx, ny = len(x), len(y)
lx = x[1] * mx
wfig = min(14., 2.5 + 2.5 * lx)                 # figure width; panel height follows the aspect ratio
fig, axs = plt.subplots(len(a.names), 1, figsize=(wfig, len(a.names) * ((wfig - 2.) * 2. / lx + 0.7)),
                        squeeze=False, constrained_layout=True)
for ax, name in zip(axs[:, 0], a.names):
    f = np.fromfile(os.path.join(a.data_dir, 'fld_%07d_%s.bin' % (a.step, name))).reshape(ny, mx)
    if a.fluct:
        f = f - f.mean(axis=1, keepdims=True)
    xp = np.append(x, lx)                      # periodic: repeat the first column
    fp = np.hstack([f, f[:, :1]])
    v = np.abs(fp).max() if (a.fluct or fp.min() < 0 < fp.max()) else None
    pc = ax.pcolormesh(xp, y, fp, shading='gouraud', cmap='RdBu_r' if v else 'viridis',
                       vmin=-v if v else None, vmax=v)
    ax.set_aspect('equal')
    ax.set_ylabel('y')
    ax.set_title(name + (" (fluctuation)" if a.fluct else ''), fontsize=10)
    fig.colorbar(pc, ax=ax, shrink=0.9)
axs[-1, 0].set_xlabel('x')
fig.savefig(a.out, dpi=150)
print('wrote', a.out)
