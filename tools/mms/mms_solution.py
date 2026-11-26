#!/usr/bin/env python3
"""
Manufactured solution for verifying solver2d with the three-equation melt condition (MMS).

    python3 tools/mms/mms_solution.py            # writes src/mms_solution.f90 (needs sympy)

Domain 0 <= x < lx (periodic), 0 <= y <= ly. Ice at y = 0 (no-slip, melt condition, meltwater velocity
v = m*), free-slip wall at y = ly with fixed scalar values theta_top, s_top. Fields (eta = y/ly, k1 = 2 pi/lx):

  prescribed at the ice:  temperature gradient  g_T(x,t) = dtheta/dy(x,0,t)
                          interface salinity    s_b(x,t)
  liquidus                theta_b = Lambda s_b + theta_f
  heat balance            m*  = St kappa_T g_T
  salt balance            g_S = ds/dy(x,0,t) = m* (s_b + sigma)/kappa_S
  scalars                 theta = theta_b + g_T phi1 + (theta_top - theta_b) phi2 + C(x,t) phi3
                          s     = s_b     + g_S phi1 + (s_top     - s_b)     phi2 + D(x,t) phi3
     phi1 = ly/pi sin(pi eta)  (phi1(0) = 0, phi1'(0) = 1, phi1(ly) = 0)
     phi2 = (1 - cos(pi eta))/2 (phi2(0) = phi2'(0) = 0, phi2(ly) = 1)
     phi3 = sin(pi eta)^2       (phi3(0) = phi3'(0) = phi3(ly) = 0)
  velocity                u = U(y,t) + dpsi/dy,  v = V(t) - dpsi/dx   (divergence-free)
     V = mean of m* (uniform: 2D continuity of the mean),  U = U0(t) sin(pi eta/2)
     psi = Psi0(x,t) cos(pi eta/2) + E(x,t) sin(pi eta) sin(pi eta/2),  dPsi0/dx = -(m* - V)
  so that u = 0 and v = m* at y = 0, and v = V, du/dy = 0, d2v/dy2 = 0 (fluctuations) at y = ly.

The profiles are trigonometric in y (not polynomial), so the compact differences are not exact. The
source terms make these fields an exact solution of the equations of solver2d:

  du/dt + (u.grad)u = -grad p + nu lap u - sum_n ri_n s_n e_g + f        (p = 0)
  ds_n/dt + u.grad s_n = kappa_n lap s_n + Q_n

  f = du/dt + (u.grad)u - nu lap u + (ri_1 theta + ri_2 s) e_g,   Q = ds/dt + u.grad s - kappa lap s

Parameters (array p, set by the code from the input): nu, kappa_T, kappa_S, ri_1, ri_2, e_g (x, y),
St, Lambda, theta_f, sigma, lx, ly, theta_top, s_top.
"""
import os
import sympy as sp

x, y, t = sp.symbols('x y t', real=True)
names = ['nu', 'kT', 'kS', 'ri1', 'ri2', 'gx', 'gy', 'St', 'Lam', 'thf', 'sig', 'lx', 'ly', 'thtop', 'stp']
P = sp.symbols(' '.join(names), real=True)
nu, kT, kS, ri1, ri2, gx, gy, St, Lam, thf, sig, lx, ly, thtop, stop = P
pi = sp.pi
PI_F = sp.Symbol('pi_', positive=True)         # pi in the generated Fortran (a parameter)
R = sp.Rational

om = 2                                         # frequency of the time dependence
k1 = 2 * pi / lx
eta = y / ly

# prescribed interface quantities
A0 = 1 + R(1, 5) * sp.sin(om * t)
A1 = R(3, 10) * sp.cos(om * t)
gT = A0 + A1 * sp.sin(k1 * x)
sb = R(1, 2) + R(1, 10) * sp.sin(om * t + R(3, 10)) * sp.cos(k1 * x)
thb = Lam * sb + thf
mst = St * kT * gT
gS = mst * (sb + sig) / kS

phi1 = ly / pi * sp.sin(pi * eta)
phi2 = (1 - sp.cos(pi * eta)) / 2
phi3 = sp.sin(pi * eta)**2
C = R(1, 5) * sp.sin(2 * k1 * x - om * t)
D = R(1, 10) * sp.cos(k1 * x + om * t)
th = thb + gT * phi1 + (thtop - thb) * phi2 + C * phi3
s = sb + gS * phi1 + (stop - sb) * phi2 + D * phi3

V = St * kT * A0                               # x-average of m*
Psi0 = St * kT * A1 * sp.cos(k1 * x) / k1      # -dPsi0/dx = m* - V
E = R(1, 2) * sp.sin(k1 * x + R(1, 2) * om * t)
psi = Psi0 * sp.cos(pi * eta / 2) + E * sp.sin(pi * eta) * sp.sin(pi * eta / 2)
U = (1 + R(1, 2) * sp.sin(om * t)) * sp.sin(pi * eta / 2)
u = U + sp.diff(psi, y)
v = V - sp.diff(psi, x)

lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)
adv = lambda f: u * sp.diff(f, x) + v * sp.diff(f, y)
b = ri1 * th + ri2 * s
fx = sp.diff(u, t) + adv(u) - nu * lap(u) + b * gx
fy = sp.diff(v, t) + adv(v) - nu * lap(v) + b * gy
qth = sp.diff(th, t) + adv(th) - kT * lap(th)
qs = sp.diff(s, t) + adv(s) - kS * lap(s)

FIELDS = dict(u=u, v=v, th=th, s=s)
SOURCES = dict(fx=fx, fy=fy, qth=qth, qs=qs)
INTERFACE = dict(mstar=mst, thb=thb, sb=sb, gT=gT, gS=gS)


def check():
    """the boundary conditions hold identically"""
    sub0, subl = {y: 0}, {y: ly}
    z = lambda e: sp.simplify(e) == 0
    ok = [z(u.subs(sub0)), z(v.subs(sub0) - mst), z(th.subs(sub0) - thb), z(s.subs(sub0) - sb),
          z(sp.diff(th, y).subs(sub0) - gT), z(sp.diff(s, y).subs(sub0) - gS),
          z(th.subs(subl) - thtop), z(s.subs(subl) - stop), z(v.subs(subl) - V),
          z(sp.diff(u, y).subs(subl)), z(sp.diff(u, x) + sp.diff(v, y))]
    return ok


def fortran(name, args_out, exprs):
    """Fortran subroutine with common-subexpression elimination"""
    exprs = [e.subs(sp.pi, PI_F) for e in exprs]
    reps, red = sp.cse(exprs, symbols=sp.numbered_symbols('w'), optimizations='basic')
    pr = lambda e: sp.fcode(e, standard=95, source_format='free').strip()
    lines = ['  subroutine %s(x,y,t,p,%s)' % (name, ','.join(args_out)),
             '    !$acc routine seq',
             '    implicit none',
             '    real(rp), intent(in)  :: x,y,t,p(nprm)',
             '    real(rp), intent(out) :: %s' % ','.join(args_out),
             '    real(rp) :: %s' % ','.join(names)]
    if reps:
        tmp = [str(r[0]) for r in reps]
        for i in range(0, len(tmp), 12):
            lines.append('    real(rp) :: %s' % ','.join(tmp[i:i+12]))
    lines.append('    ' + '; '.join('%s = p(%d)' % (n, i + 1) for i, n in enumerate(names[:8])))
    lines.append('    ' + '; '.join('%s = p(%d)' % (n, i + 9) for i, n in enumerate(names[8:])))
    for sym, e in reps:
        lines.append('    %s = %s' % (sym, pr(e)))
    for o, e in zip(args_out, red):
        lines.append('    %s = %s' % (o, pr(e)))
    lines.append('  end subroutine %s' % name)
    return '\n'.join(lines)


def write(path):
    body = [
        '!',
        '! GENERATED by tools/mms/mms_solution.py -- do not edit. Manufactured solution of solver2d with',
        '! the three-equation melt condition at y = 0 (see the generator for its definition).',
        '!   mms_exact : exact u, v, theta, s;  mms_source : source terms f_x, f_y, Q_theta, Q_s',
        '!   mms_interface : exact m*, theta_b, s_b at the ice',
        '! p(1:15) = nu, kappa_T, kappa_S, ri_1, ri_2, e_g(x), e_g(y), St, Lambda, theta_f, sigma, lx, ly,',
        '!           theta_top, s_top',
        '!',
        'module mod_mms_solution',
        '  use mod_types',
        '  implicit none',
        '  private',
        '  public :: nprm,mms_exact,mms_source,mms_interface',
        '  integer, parameter :: nprm = 15',
        '  real(rp), parameter :: pi_ = acos(-1._rp)',
        'contains',
        fortran('mms_exact', ['u', 'v', 'th', 's'], FIELDS.values()),
        '  !',
        fortran('mms_source', ['fx', 'fy', 'qth', 'qs'], SOURCES.values()),
        '  !',
        fortran('mms_interface', ['mstar', 'thb', 'sb'], [mst, thb, sb]),
        'end module mod_mms_solution',
        '']
    open(path, 'w').write('\n'.join(body))


if __name__ == '__main__':
    ok = check()
    print('boundary conditions and continuity hold identically:', all(ok), ok)
    if not all(ok):
        raise SystemExit('the manufactured solution does not satisfy the conditions')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'mms_solution.f90')
    write(os.path.normpath(out))
    print('wrote', os.path.normpath(out))
