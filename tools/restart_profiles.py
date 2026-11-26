#!/usr/bin/env python3
"""
Write the wall-normal profiles of the scalars (e.g. temperature T and salinity S) from a solver2d restart
file to an ASCII file.

    python3 tools/restart_profiles.py data/restart [-o profiles.dat] [--names T S] [--velocity]
                                      [--grid-file grid.in]

The restart file (restart.hdr + restart.bin) holds u, v and the scalars as Fourier coefficients
(complex, global (nx/2, ny) arrays). The profile written is the x-average, i.e. the kx = 0 coefficient,
at the ny wall-normal grid points. The grid is rebuilt from the header (grid type, alpha_s, ny and the
wall positions), as in the code; for grid_type = 'file' give the grid file with --grid-file.

Output: a header (time, step, grid) and the columns  y  [U]  T  S  (or s1 s2 ... for other numbers of
scalars, or the names given with --names). --velocity adds the mean streamwise velocity U(y).
Needs only numpy.
"""
import argparse
import numpy as np


def read_header(fname):
    lines = [l.split() for l in open(fname + '.hdr').read().splitlines()]
    nx, ny, nscal = (int(v) for v in lines[1][:3])
    lx = float(lines[2][0])
    ybot, ytop = (float(lines[2][1]), float(lines[2][2])) if len(lines[2]) >= 3 else (-1.0, 1.0)
    time, istep, dt = float(lines[3][0]), int(lines[3][1]), float(lines[3][2])
    gtype = lines[4][0] if len(lines) > 4 else 'tanh'
    alpha = float(lines[4][1]) if len(lines) > 4 else 2.0
    return dict(nx=nx, ny=ny, nscal=nscal, lx=lx, ybot=ybot, ytop=ytop, time=time, istep=istep, dt=dt,
                grid_type=gtype, grid_alpha=alpha)


def grid(h, grid_file=None):
    """wall-normal grid of solver2d (src/compact.f90, init_grid)"""
    ny, a = h['ny'], h['grid_alpha']
    xi = np.arange(ny) / (ny - 1)
    g = h['grid_type']
    if g == 'tanh':
        eta = np.tanh(a * (xi - 0.5)) / np.tanh(0.5 * a)
    elif g == 'tanh_top':
        eta = -1 + 2 * np.tanh(0.5 * a * xi) / np.tanh(0.5 * a)
    elif g == 'tanh_bot':
        eta = 1 - 2 * np.tanh(0.5 * a * xi[::-1]) / np.tanh(0.5 * a)
    elif g == 'uniform':
        eta = -1 + 2 * xi
    elif g == 'file':
        if grid_file is None:
            raise SystemExit("grid_type = 'file': give the grid file with --grid-file")
        y = np.loadtxt(grid_file)[:ny]
        y[0], y[-1] = h['ybot'], h['ytop']
        return y
    else:
        raise SystemExit('unknown grid type ' + g)
    y = 0.5 * (h['ybot'] + h['ytop']) + 0.5 * (h['ytop'] - h['ybot']) * eta
    y[0], y[-1] = h['ybot'], h['ytop']
    return y


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('restart', help='restart file name without extension, e.g. data/restart')
ap.add_argument('-o', '--out', default='profiles.dat')
ap.add_argument('--names', nargs='+', help='names of the scalar columns (default: T S for two scalars)')
ap.add_argument('--velocity', action='store_true', help='also write the mean velocity U(y)')
ap.add_argument('--grid-file', default=None)
a = ap.parse_args()

h = read_header(a.restart)
nxh, ny, ns = h['nx'] // 2, h['ny'], h['nscal']
data = np.fromfile(a.restart + '.bin', dtype=np.complex128)
nvar = data.size // (nxh * ny)
if nvar * nxh * ny != data.size or nvar != 2 + ns:
    raise SystemExit('restart.bin has %d values; expected (2 + nscal) x nx/2 x ny = %d x %d x %d'
                     % (data.size, 2 + ns, nxh, ny))
f = data.reshape(nvar, ny, nxh)          # Fortran (nx/2, ny) per field
mean = f[:, :, 0].real                   # kx = 0: x-average

names = a.names if a.names else (['T', 'S'] if ns == 2 else ['s%d' % (n + 1) for n in range(ns)])
if len(names) != ns:
    raise SystemExit('--names: %d names for %d scalars' % (len(names), ns))
y = grid(h, a.grid_file)
cols, labels = [y], ['y']
if a.velocity:
    cols.append(mean[0]); labels.append('U')
for n in range(ns):
    cols.append(mean[2 + n]); labels.append(names[n])

with open(a.out, 'w') as fo:
    fo.write('# x-averaged profiles from %s.bin: t = %.10g, step %d\n' % (a.restart, h['time'], h['istep']))
    fo.write('# grid %s (alpha_s = %g), ny = %d, walls at y = %g and %g\n'
             % (h['grid_type'], h['grid_alpha'], ny, h['ybot'], h['ytop']))
    fo.write('#' + ''.join('%23s' % l for l in labels)[1:] + '\n')
    np.savetxt(fo, np.column_stack(cols), fmt='%23.15e', delimiter='')
print('wrote %s: %d points, columns %s (t = %g)' % (a.out, ny, ' '.join(labels), h['time']))
