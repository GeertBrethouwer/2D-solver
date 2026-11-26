#!/usr/bin/env python3
"""
MMS convergence study of solver2d with the three-equation melt condition (input.nml in this directory).

    python3 convergence.py [--exe ../../build/solver2d] [--mpirun "mpirun -np 1"] [--jobs 4] [--xdiff explicit]

Space: ny = 17, 33, 65, 129, 257 at dt = 5e-4 (the time error is negligible), t = 0..0.5. Errors against the
       exact solution, from data/mms_error.dat (max over the physical grid; at the ice: m*, theta_b, s_b).
Time:  dt = 0.016 ... 0.002 on ny = 65, t = 0..0.48, against a run with dt = 2.5e-4 on the same grid (the
       spatial error cancels). Max differences of the Fourier coefficients of u, v, theta, s and of the
       wall values v(y=0) = m*.
Prints the errors and the observed orders log2(e(h)/e(h/2)) and writes mms_convergence.png (matplotlib).
"""
import argparse, os, re, shlex, subprocess
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--exe', default='../../build/solver2d')
ap.add_argument('--mpirun', default='mpirun -np 1')
ap.add_argument('--jobs', type=int, default=4)
ap.add_argument('--xdiff', default='explicit', choices=['explicit', 'implicit'])
ap.add_argument('--grad', default='compact', choices=['compact', 'explicit'],
                help='melt_grad: wall gradient of the melt condition')
ap.add_argument('--rundir', default='runs')
ap.add_argument('--noplot', action='store_true')
ap.add_argument('--fig', default=None, help='figure file (default mms_convergence_<xdiff>_<grad>.png; .pdf works)')
ap.add_argument('--norun', action='store_true', help='only evaluate existing runs')
a = ap.parse_args()
exe = os.path.abspath(a.exe)
base = open('input.nml').read()
tag = '%s_%s' % (a.xdiff, a.grad)
NY_SPACE, DT_SPACE, T_SPACE = [17, 33, 65, 129, 257], 5e-4, 0.5
DT_TIME, DT_REF, T_TIME, NY_TIME = [1.6e-2, 8e-3, 4e-3, 2e-3], 2.5e-4, 0.48, 65


def make(name, ny, dt, tend):
    d = os.path.join(a.rundir, tag, name)
    os.makedirs(d, exist_ok=True)
    n = int(round(tend / dt))
    txt = re.sub(r'ny\s*=\s*\d+', 'ny = %d' % ny, base)
    txt = re.sub(r'nstep\s*=\s*\d+, fixed_dt = .true., dt = [0-9.eE+-]+',
                 'nstep = %d, fixed_dt = .true., dt = %.6e' % (n, dt), txt)
    txt = re.sub(r'iout_log\s*=\s*\d+', 'iout_log = %d' % max(n // 5, 1), txt)
    txt = re.sub(r'iout_restart\s*=\s*\d+', 'iout_restart = %d' % n, txt)
    txt = re.sub(r"melt_xdiff\s*=\s*'\w+'", "melt_xdiff = '%s', melt_grad = '%s'" % (a.xdiff, a.grad), txt)
    open(os.path.join(d, 'input.nml'), 'w').write(txt)
    return d


runs = [make('space_ny%d' % ny, ny, DT_SPACE, T_SPACE) for ny in NY_SPACE] + \
       [make('time_dt%g' % dt, NY_TIME, dt, T_TIME) for dt in DT_TIME + [DT_REF]]
procs = []
for d in ([] if a.norun else runs):                                   # run in parallel, at most --jobs at a time
    while len([p for p in procs if p.poll() is None]) >= a.jobs:
        procs[[p.poll() is None for p in procs].index(True)].wait()
    subprocess.run('rm -rf ' + os.path.join(d, 'data'), shell=True)
    procs.append(subprocess.Popen(shlex.split(a.mpirun) + [exe], cwd=d, stdout=open(os.path.join(d, 'out.txt'), 'w'),
                                  stderr=subprocess.STDOUT))
for p in procs:
    p.wait()

order = lambda e: ['    -'] + ['%5.2f' % np.log2(e[i - 1] / e[i]) if e[i] > 0 else '    -' for i in range(1, len(e))]
print('MMS convergence, melt_xdiff = %s, melt_grad = %s' % (a.xdiff, a.grad))
print('\nspace (dt = %g, t = %g): max errors against the exact solution, observed order' % (DT_SPACE, T_SPACE))
names = ['u', 'v', 'theta', 's', 'm*(wall)', 'theta_b', 's_b']
E = np.array([np.loadtxt(os.path.join(a.rundir, tag, 'space_ny%d' % ny, 'data', 'mms_error.dat'))[-1, 1:]
              for ny in NY_SPACE])
print('   ny' + ''.join('%19s' % n for n in names))
O = [order(E[:, j]) for j in range(E.shape[1])]
for i, ny in enumerate(NY_SPACE):
    print('%5d' % ny + ''.join('%11.3e (%5s)' % (E[i, j], O[j][i].strip()) for j in range(E.shape[1])))


def load(d, ny):
    r = np.fromfile(os.path.join(d, 'data', 'restart.bin'), dtype=np.complex128).reshape(4, ny, -1)
    return r


print('\ntime (ny = %d, t = %g): max differences to the run with dt = %g, observed order' % (NY_TIME, T_TIME, DT_REF))
ref = load(os.path.join(a.rundir, tag, 'time_dt%g' % DT_REF), NY_TIME)
Et = []
for dt in DT_TIME:
    r = load(os.path.join(a.rundir, tag, 'time_dt%g' % dt), NY_TIME)
    Et.append([abs(r[k] - ref[k]).max() for k in range(4)] + [abs(r[1][0] - ref[1][0]).max()])
Et = np.array(Et)
tn = ['u', 'v', 'theta', 's', 'm*(wall)']
print('       dt' + ''.join('%19s' % n for n in tn))
O = [order(Et[:, j]) for j in range(Et.shape[1])]
for i, dt in enumerate(DT_TIME):
    print('%9.2e' % dt + ''.join('%11.3e (%5s)' % (Et[i, j], O[j][i].strip()) for j in range(Et.shape[1])))

if not a.noplot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    h = 1.0 / (np.array(NY_SPACE) - 1)
    for j, n in enumerate(names):
        ax[0].loglog(h, E[:, j], 'o-', label=n)
    for p, c in [(3, 'k'), (4, 'gray')]:
        ax[0].loglog(h, E[2, 0] * (h / h[2])**p, ls='--', color=c, lw=0.8, label='order %d' % p)
    ax[0].set_xlabel('mean grid spacing 1/(ny-1)'); ax[0].set_ylabel('max error'); ax[0].legend(fontsize=8, ncol=2)
    ax[0].set_title('space (dt = %g)' % DT_SPACE)
    dts = np.array(DT_TIME)
    for j, n in enumerate(tn):
        ax[1].loglog(dts, Et[:, j], 'o-', label=n)
    ax[1].loglog(dts, Et[-1, 3] * (dts / dts[-1])**2, 'k--', lw=0.8, label='order 2')
    ax[1].set_xlabel('dt'); ax[1].set_ylabel('max difference to dt = %g' % DT_REF); ax[1].legend(fontsize=8)
    ax[1].set_title('time (ny = %d)' % NY_TIME)
    fig.suptitle('solver2d MMS with the melt condition (melt_xdiff = %s, melt_grad = %s)' % (a.xdiff, a.grad))
    fig.tight_layout()
    fname = a.fig if a.fig else 'mms_convergence_%s.png' % tag
    fig.savefig(fname, dpi=130)
    print('\nwrote %s' % fname)
