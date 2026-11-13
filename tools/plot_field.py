#!/usr/bin/env python3
"""
Plot 2D fields written by solver2d (iout_field > 0): data/fld_NNNNNNN_<name>.bin on the dealiased
mx x ny grid (Fortran order (x,y), float64), with data/grid_x.bin and data/grid_y.bin.

    python3 tools/plot_field.py data 8000 u v s1 s2 [options] [-o fields.png]

Colour maps
  --cmap NAME       colour map of all fields
  --scmap NAME      colour map of the scalars (s1, s2, ...); default: --cmap if given
  --cmap-dir DIR    directory of the Scientific Colour Maps (Crameri, doi:10.5281/zenodo.1243862),
                    default $SCM_DIR or /scratch/geert/matlab_colours
  NAME is a Scientific Colour Map (e.g. acton, batlow, vik, roma), read from DIR/NAME/NAME.txt
  (or DIR/NAME.txt, or the .mat/.cpt files of the distribution), a path to such a file, or a
  matplotlib colour map (e.g. RdBu_r, viridis). Append _r to reverse a colour map (acton_r).
  Defaults without --cmap/--scmap: RdBu_r for scalars and for fields with both signs, else viridis.

Colour range (values outside the range are saturated; arrows on the colour bar)
  --srange MIN MAX       range of all scalars
  --clim NAME MIN MAX    range of field NAME (repeatable; overrides --srange), e.g. --clim s2 0 0.5
  Automatic otherwise:
    scalars     : smallest to largest wall value, i.e. the values at y = 0 and y = ly (with Dirichlet
                  walls the two prescribed values); field min..max if both walls have the same value
    other fields: symmetric about zero (-max|f|..max|f|) if the field has both signs, else min..max
  --fluct subtracts the x-average from each field; the automatic range is then symmetric about zero.

Needs numpy and matplotlib (scipy only to read .mat colour-map files).
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def read_rgb(path, name):
    """RGB table (N x 3, values 0..1) from a Scientific Colour Map file (.txt, .mat or .cpt)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mat':
        from scipy.io import loadmat               # only needed for .mat files
        d = loadmat(path)
        rgb = d.get(name)
        if rgb is None:
            rgb = next(v for k, v in d.items() if not k.startswith('__')
                       and isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] >= 3)
    elif ext == '.cpt':
        rows, last = [], None
        for line in open(path):
            t = line.split()
            if not t or t[0][0] in '#BFN':
                continue
            rows.append([float(c) for c in t[1:4]])
            last = t
        rows.append([float(c) for c in last[5:8]])  # upper colour of the last segment
        rgb = np.array(rows)
    else:
        rgb = np.loadtxt(path)
    rgb = np.asarray(rgb, dtype=float)[:, :3]
    if rgb.max() > 1.0:                            # 0..255 tables
        rgb = rgb / 255.0
    return rgb


def get_cmap(name, cmap_dir):
    """Colour map from a Scientific Colour Map name/file or a matplotlib name; '_r' reverses."""
    rev = name.endswith('_r')
    base = name[:-2] if rev else name
    if os.path.isfile(base):
        cands = [base]
        base = os.path.splitext(os.path.basename(base))[0]
    else:
        cands = [os.path.join(cmap_dir, base, base + e) for e in ('.txt', '.mat', '.cpt')] + \
                [os.path.join(cmap_dir, base + e) for e in ('.txt', '.mat', '.cpt')]
    for c in cands:
        if os.path.isfile(c):
            try:
                cm = ListedColormap(read_rgb(c, base), name=base)
            except ImportError:
                continue                           # .mat without scipy: try the next file
            return cm.reversed() if rev else cm
    try:                                           # matplotlib colour map (incl. its own _r)
        return plt.get_cmap(name)
    except ValueError:
        raise SystemExit('colour map %s not found: no matplotlib colour map of that name, and none of\n  %s'
                         % (name, '\n  '.join(cands)))


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('data_dir')
ap.add_argument('step', type=int)
ap.add_argument('names', nargs='+')
ap.add_argument('--fluct', action='store_true')
ap.add_argument('--cmap', default=None)
ap.add_argument('--scmap', default=None)
ap.add_argument('--cmap-dir', default=os.environ.get('SCM_DIR', '/scratch/geert/matlab_colours'))
ap.add_argument('--srange', nargs=2, type=float, metavar=('MIN', 'MAX'))
ap.add_argument('--clim', nargs=3, action='append', default=[], metavar=('NAME', 'MIN', 'MAX'))
ap.add_argument('-o', '--out', default='fields.png')
a = ap.parse_args()
clim = {n: (float(lo), float(hi)) for n, lo, hi in a.clim}
for lo, hi in list(clim.values()) + ([tuple(a.srange)] if a.srange else []):
    if not lo < hi:
        raise SystemExit('colour range: MIN must be smaller than MAX')
cmap_all = get_cmap(a.cmap, a.cmap_dir) if a.cmap else None
cmap_s = get_cmap(a.scmap, a.cmap_dir) if a.scmap else cmap_all

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
    scalar = name.startswith('s')
    # automatic range and default colour map
    if a.fluct or (not scalar and fp.min() < 0 < fp.max()):
        v = np.abs(fp).max()
        vmin, vmax, cmap = -v, v, 'RdBu_r'
    elif scalar:
        walls = np.concatenate([f[0], f[-1]])          # values at y = 0 and y = ly
        vmin, vmax, cmap = walls.min(), walls.max(), 'RdBu_r'
        if vmax - vmin <= 1e-12 * max(abs(vmax), 1.):  # same value at both walls
            vmin, vmax = fp.min(), fp.max()
    else:
        vmin, vmax, cmap = fp.min(), fp.max(), 'viridis'
    # user choices
    if name in clim:
        vmin, vmax = clim[name]
    elif scalar and a.srange:
        vmin, vmax = a.srange
    if scalar and cmap_s is not None:
        cmap = cmap_s
    elif not scalar and cmap_all is not None:
        cmap = cmap_all
    lo, hi = fp.min() < vmin, fp.max() > vmax
    extend = 'both' if lo and hi else 'min' if lo else 'max' if hi else 'neither'
    pc = ax.pcolormesh(xp, y, fp, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_ylabel('y')
    ax.set_title(name + (" (fluctuation)" if a.fluct else ''), fontsize=10)
    fig.colorbar(pc, ax=ax, shrink=0.9, extend=extend)
axs[-1, 0].set_xlabel('x')
fig.savefig(a.out, dpi=150)
print('wrote', a.out)
