#!/usr/bin/env python3
"""
Compare the melt-validation runs (case1/, case2/) with the similarity solution of the report
"Coupled diffusion of temperature and concentration in a semi-infinite domain with fixed and moving
interfaces" (Sec. 3), and draw its Figures 1-5.

    cd examples/melt_validation
    (cd case1 && mpirun -np 1 ../../../build/solver2d)
    (cd case2 && mpirun -np 1 ../../../build/solver2d)
    python3 compare.py [-o figures]

Theory (interface z = a(t) = 2 lambda sqrt(k t), interface-attached coordinate z' = z - a = y):
    lambda from  T_inf + lambda S sqrt(pi) e^{lambda^2} erfc(lambda)
                 + Gamma C_inf/(1 - lambda sqrt(pi Le) e^{lambda^2 Le} erfc(lambda sqrt(Le))) = 0  (27)
    C_w = C_inf/(1 - lambda sqrt(pi Le) e^{lambda^2 Le} erfc(lambda sqrt(Le))),  T_w = -Gamma C_w (26)
    T(z',t) = T_inf + (T_w - T_inf) erfc(lambda + z'/(2 sqrt(k t)))/erfc(lambda)                (21)
    C(z',t) = C_inf + (C_w - C_inf) erfc(lambda sqrt(Le) + z'/(2 sqrt(D t)))/erfc(lambda sqrt(Le))(22)
    melt speed V_melt = -da/dt = -lambda sqrt(k/t)                                              (17)
In solver2d: V_melt = m* (log.dat column 15), C_w = s_b (column 17), T_w = theta_b (column 16).
Needs numpy, scipy, matplotlib.
"""
import argparse, os, re
import numpy as np
from scipy.special import erfc, erfcx
from scipy.optimize import brentq

CASES = {'case1': dict(k=1/1200, Le=10.0, S=10.0, G=8.0, Tinf=0.0, Cinf=1.0),
         'case2': dict(k=1/1200, Le=100.0, S=1.0, G=0.1, Tinf=1.0, Cinf=1.0)}


def theory(k, Le, S, G, Tinf, Cinf):
    """lambda, C_w, T_w from (25)-(27); erfcx(x) = exp(x^2) erfc(x) avoids overflow"""
    cw = lambda l: Cinf / (1 - l * np.sqrt(np.pi * Le) * erfcx(l * np.sqrt(Le)))
    f = lambda l: Tinf + l * S * np.sqrt(np.pi) * erfcx(l) + G * cw(l)
    lam = brentq(f, -5, -1e-12, xtol=1e-15)
    return lam, cw(lam), -G * cw(lam)


def profiles(z, t, k, Le, lam, Tw, Cw, Tinf, Cinf):
    D = k / Le
    T = Tinf + (Tw - Tinf) * erfc(lam + z / (2 * np.sqrt(k * t))) / erfc(lam)
    C = Cinf + (Cw - Cinf) * erfc(lam * np.sqrt(Le) + z / (2 * np.sqrt(D * t))) / erfc(lam * np.sqrt(Le))
    return T, C


def nml(path, key, default=None):
    """first value of key in a namelist file (enough for the simple inputs here)"""
    txt = re.sub(r'!.*', '', open(path).read())
    m = re.search(r'\b' + re.escape(key) + r'\s*=\s*([^,/\n]+)', txt)
    return m.group(1).strip().strip("'") if m else default


def grid(d):
    """wall-normal grid of solver2d ('tanh_bot', 'tanh_top', 'tanh' or 'uniform'), y in [0, ly]"""
    inp = os.path.join(d, 'input.nml')
    ny, ly = int(nml(inp, 'ny')), float(nml(inp, 'ly', 2.0))
    a = float(nml(inp, 'grid_alpha', 2.0))
    gt = nml(inp, 'grid_type', 'auto')
    if gt == 'auto':
        gt = 'tanh_top' if nml(inp, 'bc_bot', 'noslip') == 'freeslip' else \
             'tanh_bot' if nml(inp, 'bc_top', 'noslip') == 'freeslip' else 'tanh'
    xi = np.arange(ny) / (ny - 1)
    if gt == 'tanh':
        eta = np.tanh(a * (xi - 0.5)) / np.tanh(0.5 * a)
    elif gt == 'tanh_top':
        eta = -1 + 2 * np.tanh(0.5 * a * xi) / np.tanh(0.5 * a)
    elif gt == 'tanh_bot':
        eta = 1 - 2 * np.tanh(0.5 * a * xi[::-1]) / np.tanh(0.5 * a)
    else:
        eta = -1 + 2 * xi
    return 0.5 * ly * (1 + eta)


def load(d):
    log = np.loadtxt(os.path.join(d, 'data', 'log.dat'))
    y = grid(d)
    ny = len(y)
    nxh = int(nml(os.path.join(d, 'input.nml'), 'nx')) // 2
    r = np.fromfile(os.path.join(d, 'data', 'restart.bin'), dtype=np.complex128).reshape(-1, ny, nxh)
    hdr = open(os.path.join(d, 'data', 'restart.hdr')).read().split('\n')
    tend = float(hdr[3].split()[0])
    return log, y, r[2][:, 0].real, r[3][:, 0].real, tend     # T, C: mean (kx = 0) profiles


ap = argparse.ArgumentParser()
ap.add_argument('-o', '--outdir', default='figures')
ap.add_argument('--noplot', action='store_true')
a = ap.parse_args()
res = {}
for c, p in CASES.items():
    if not os.path.isfile(os.path.join(c, 'data', 'log.dat')):
        print('%s: no data (run solver2d in %s/ first)' % (c, c))
        continue
    lam, Cw, Tw = theory(**p)
    log, y, T, C, tend = load(c)
    t, vm, Tb, Cb = log[1:, 1], log[1:, 14], log[1:, 15], log[1:, 16]
    Tex, Cex = profiles(y, tend, p['k'], p['Le'], lam, Tw, Cw, p['Tinf'], p['Cinf'])
    vex = -lam * np.sqrt(p['k'] / t)
    late = t > 0.1
    print('== %s: Le = %g, S = %g, Gamma = %g, T_inf = %g, C_inf = %g' % (c, p['Le'], p['S'], p['G'], p['Tinf'], p['Cinf']))
    print('   theory    : lambda = %.8f, C_w = %.8g, T_w = %.8g' % (lam, Cw, Tw))
    print('   simulation: t = %.4f, C_w = %.8g, T_w = %.8g' % (t[-1], Cb[-1], Tb[-1]))
    print('   C_w error %.2e (abs.), T_w error %.2e (abs.)' % (abs(Cb[-1] - Cw), abs(Tb[-1] - Tw)))
    print('   melt speed: rel. error at t_end %.2e, max rel. error for t > 0.1 %.2e'
          % (abs(vm[-1] - vex[-1]) / vex[-1], (abs(vm - vex) / vex)[late].max()))
    print('   profiles at t_end: max |T - T_theory| = %.2e, max |C - C_theory| = %.2e'
          % (abs(T - Tex).max(), abs(C - Cex).max()))
    res[c] = (p, lam, Cw, Tw, t, vm, Tb, Cb, y, T, C, Tex, Cex, tend)

if a.noplot or not res:
    raise SystemExit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.makedirs(a.outdir, exist_ok=True)
plt.rcParams.update({'font.size': 13})
blue, red, green = '#3d7db5', '#e0312b', '#4caf50'


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, name), dpi=150)
    plt.close(fig)
    print('wrote', os.path.join(a.outdir, name))


if 'case1' in res:
    p, lam, Cw, Tw, t, vm, Tb, Cb, y, T, C, Tex, Cex, tend = res['case1']
    fig, ax = plt.subplots(figsize=(5, 4.2))                     # Figure 1
    ax.semilogx(t, Cb, color=blue, label='simulation')
    ax.axhline(Cw, color=red, ls='--', label='theory')
    vis = Cb[t >= 1e-3]
    pad = 0.05 * (max(vis.max(), Cw) - min(vis.min(), Cw))
    ax.set_xlabel('$t$'); ax.set_ylabel('$C_w$'); ax.set_xlim(1e-3, t[-1])
    ax.set_ylim(min(vis.min(), Cw) - pad, max(vis.max(), Cw) + pad); ax.legend()
    save(fig, 'fig1_case1_Cw.png')
    fig, ax = plt.subplots(figsize=(5, 4.2))                     # Figure 2
    ax.plot(y, T, color=blue, label='simulation')
    ax.plot(y, Tex, color=red, ls='--', label='theory (21)')
    ax.set_xlabel('$y$'); ax.set_ylabel('$T$'); ax.set_xlim(0, y[-1]); ax.legend()
    save(fig, 'fig2_case1_T.png')
    fig, ax = plt.subplots(figsize=(5, 4.2))                     # Figure 3
    ax.loglog(t, vm, color=blue, label='simulation')
    tt = t[t >= 1e-3]
    ax.loglog(tt, -lam * np.sqrt(p['k'] / tt), color=red, ls='-.', label='theory (17)')
    ax.set_xlabel('$t$'); ax.set_ylabel('$V_{melt}$'); ax.set_xlim(1e-4, t[-1]); ax.legend()
    save(fig, 'fig3_case1_Vmelt.png')
if 'case2' in res:
    p, lam, Cw, Tw, t, vm, Tb, Cb, y, T, C, Tex, Cex, tend = res['case2']
    fig, ax = plt.subplots(figsize=(5, 4.2))                     # Figure 4
    ax.plot(y, T, color=blue, label='$T$ simulation')
    ax.plot(y, Tex, color=red, ls='--', label='$T$ theory (21)')
    ax.plot(y, C, color=green, label='$C$ simulation')
    ax.plot(y, Cex, color='k', ls=':', label='$C$ theory (22)')
    ax.set_xlabel('$y$'); ax.set_ylabel('$T, C$'); ax.set_xlim(0, y[-1]); ax.legend(loc='lower right')
    save(fig, 'fig4_case2_T_C.png')
    fig, ax = plt.subplots(figsize=(5, 4.2))                     # Figure 5
    ax.loglog(t, vm, color=blue, label='simulation')
    tt = t[t >= 1e-3]
    ax.loglog(tt, -lam * np.sqrt(p['k'] / tt), color=red, ls='--', label='theory (17)')
    ax.set_xlabel('$t$'); ax.set_ylabel('$V_{melt}$'); ax.set_xlim(1e-4, t[-1]); ax.legend()
    save(fig, 'fig5_case2_Vmelt.png')
