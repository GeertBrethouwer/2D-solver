#!/usr/bin/env python3
# run the Rayleigh-Benard onset case at three total Richardson numbers and estimate Ra_c
#   python3 check.py [path/to/solver2d] [mpirun command]
import os, re, subprocess, sys
import numpy as np
exe = sys.argv[1] if len(sys.argv) > 1 else '../../build/solver2d'
launcher = sys.argv[2].split() if len(sys.argv) > 2 else ['mpirun', '-np', '1']
ric = 1707.76 / (16 * 100.**2 * 1.0)
base = open('input.nml').read()
facs, sig = [0.9, 1.0, 1.1], []
for f in facs:
    d = 'run_%.1f' % f
    os.makedirs(d, exist_ok=True)
    txt = re.sub(r'ri\(1\)\s*=\s*[0-9.eE+-]+', 'ri(1) = %.10e' % (0.7 * f * ric), base)
    txt = re.sub(r'ri\(2\)\s*=\s*[0-9.eE+-]+', 'ri(2) = %.10e' % (0.3 * f * ric), txt)
    open(os.path.join(d, 'input.nml'), 'w').write(txt)
    subprocess.run(launcher + [os.path.abspath(exe)], cwd=d, stdout=subprocess.DEVNULL, check=True)
    a = np.loadtxt(os.path.join(d, 'data', 'log.dat'))
    m = a[:, 1] > 800
    sig.append(np.polyfit(a[m, 1], np.log(a[m, 9]), 1)[0] / 2)
    print('(ri1+ri2)/ri_c = %.1f   growth rate = %+.4e' % (f, sig[-1]))
p = np.polyfit(facs, sig, 2)
r = [x.real for x in np.roots(p) if 0.5 < x.real < 1.5][0]
print('Ra_c = %.2f   (linear theory 1707.76)' % (1707.76 * r))
