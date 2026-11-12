#!/usr/bin/env python3
"""
Plot 2D fields written by solver2d (iout_field > 0): data/fld_NNNNNNN_<name>.bin on the dealiased
mx x ny grid (Fortran order (x,y), float64), with data/grid_x.bin and data/grid_y.bin.

    python3 tools/plot_field.py data 8000 u v s1 s2 [--fluct] [-o fields.png]

Colour range:
  scalars (s1, s2, ...): from the smallest to the largest wall value, i.e. the values at y = 0 and y = ly
                         (with Dirichlet walls the two prescribed values); values outside are saturated
                         (arrows on the colour bar). If both walls have the same value: field min..max.
  other fields         : symmetric about zero (-max|f|..max|f|) if the field has both signs, else min..max.
--fluct subtracts the x-average from each field; the colour range is then symmetric about zero.
Needs numpy and matplotlib.
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
ly = y[-1] - y[0]
wfig = min(14., 2.5 + 2.5 * lx)                 # figure width; panel height follows the aspect ratio
fig, axs = plt.subplots(len(a.names), 1, figsize=(wfig, len(a.names) * ((wfig - 2.) * ly / lx + 0.7)),
                        squeeze=False, constrained_layout=True)
for ax, name in zip(axs[:, 0], a.names):
    f = np.fromfile(os.path.join(a.data_dir, 'fld_%07d_%s.bin' % (a.step, name))).reshape(ny, mx)
    if a.fluct:
        f = f - f.mean(axis=1, keepdims=True)
    xp = np.append(x, lx)                      # periodic: repeat the first column
    fp = np.hstack([f, f[:, :1]])
    extend = 'neither'
    if a.fluct or (not name.startswith('s') and fp.min() < 0 < fp.max()):
        v = np.abs(fp).max()
        vmin, vmax, cmap = -v, v, 'RdBu_r'
    elif name.startswith('s'):
        walls = np.concatenate([f[0], f[-1]])          # values at y = 0 and y = ly
        vmin, vmax, cmap = walls.min(), walls.max(), 'RdBu_r'
        if vmax - vmin <= 1e-12 * max(abs(vmax), 1.):  # same value at both walls
            vmin, vmax = fp.min(), fp.max()
        lo, hi = fp.min() < vmin, fp.max() > vmax
        extend = 'both' if lo and hi else 'min' if lo else 'max' if hi else 'neither'
    else:
        vmin, vmax, cmap = fp.min(), fp.max(), 'viridis'
    pc = ax.pcolormesh(xp, y, fp, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_ylabel('y')
    ax.set_title(name + (" (fluctuation)" if a.fluct else ''), fontsize=10)
    fig.colorbar(pc, ax=ax, shrink=0.9, extend=extend)
axs[-1, 0].set_xlabel('x')
fig.savefig(a.out, dpi=150)
print('wrote', a.out)
