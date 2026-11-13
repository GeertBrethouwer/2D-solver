# 2D solver — DNS of two-dimensional channel flow with active scalars (CPU and Grace Hopper GPUs)

`solver2d` solves the incompressible Navier–Stokes equations in a two-dimensional plane channel
(x periodic with length `lx`, walls at y = 0 and y = `ly`), together with active scalars (two by default, up to 16) that act on the flow
through a Boussinesq buoyancy term. It is the two-dimensional counterpart of the velocity–vorticity option
(`ns_method = 'vv'`) of [HiCHAN](https://github.com/GeertBrethouwer/hichan). The numerics, input format,
output files and GPU design are the same as in HiCHAN, with the spanwise direction removed.

* **x:** Fourier series, pseudo-spectral nonlinear terms, 3/2-rule dealiasing.
* **y:** collocated compact finite differences on an arbitrary (e.g. tanh-stretched) grid, with the (3-4-3)
  fourth-order or (3-5-6-5-3) sixth-order schemes of Lenaers, Schlatter, Brethouwer & Johansson,
  *J. Comput. Phys.* 272 (2014) 108–126. The coefficients are computed directly on the non-uniform grid.
* **Velocity–vorticity formulation** (Kim, Moin & Moser 1987; Lluesma-Rodríguez, Álcantara-Ávila,
  Pérez-Quiles & Hoyas, *Mathematics* 9 (2021) 756), without the pressure.
* **Time:** three-substep RK3 of Spalart, Moser & Rogers (1991). The viscous terms are implicit; scalar
  diffusion uses the θ-method.
* **Scalars:** any number (default 2), passive or active, with Dirichlet or Neumann walls, sources, and gravity in
  any direction in the x–y plane. All scalars are solved in one batched GPU kernel.
* **Walls:** at y = 0 and y = `ly` (box height, default 2). Each is no-slip or free-slip (`bc_bot` for y = 0,
  `bc_top` for y = `ly`; at most one free-slip), e.g. for an open channel or water under an ice layer. The
  default grid is refined at every no-slip wall: at both walls, or only at the no-slip one.
  "bot"/"top" in the input always mean y = 0 / y = `ly`, whatever the direction of gravity.
* **Melting ice:** the three-equation melt condition (liquidus, heat and salt balances) at an ice wall at
  y = 0, with the meltwater velocity at the interface (`&melt_model`, see *Melting ice* below).
* **Parallelisation:** MPI with one rank per GPU, OpenACC kernels and cuFFT, made for one NSC Arrhenius node
  (4 × NVIDIA GH200). The same source runs on CPUs with FFTW.

## Equations

With `N = u × ω + buoyancy + mean pressure gradient` (rotational form; or the advective form), the code evolves
for every streamwise wavenumber `kx ≠ 0`:

```
∂φ/∂t = h_v + (1/Re) (D2 − kx²) φ ,      (D2 − kx²) v = φ ,      v = dv/dy = 0 at the walls
h_v   = −kx² N_y − i kx D1 N_x
u     = i D1 v / kx                (continuity, i kx u + D1 v = 0)
```

and for `kx = 0` the mean flow `∂U/∂t = N_x,00 + (1/Re) D2 U` (U = 0 at the walls), with a constant flow rate
or a constant pressure gradient. In 2D there is no wall-normal vorticity (ω_y = ∂u/∂z − ∂w/∂x = 0), so φ
alone carries the dynamics. It is the x-derivative of the spanwise vorticity, φ = ∂ω_z/∂x.

**Free-slip wall** (`bc_bot = 'freeslip'` at y = 0, or `bc_top = 'freeslip'` at y = `ly`; at most one):
* **The condition:** v = 0 and ∂u/∂y = 0 at that wall. Since u = i D1 v/kx, the second condition becomes
  ∂²v/∂y² = 0 there. It replaces dv/dy = 0 in the same 2×2 influence-matrix superposition, so the cost is
  unchanged.
* **The wall operator:** a 7-point one-sided second derivative (`vv_neumann = 'explicit'`), or the first/last
  row of the compact D2 (`'compact'`).
* **The mean flow:** dU/dy = 0 at that wall, imposed by superposition with the homogeneous solution U = 1 at
  that wall, using the compact D1 row.
* **Laminar start:** the open-channel profile U = U_max(1 − ζ²), with ζ = distance from the free-slip wall/`ly`.
  With CFR, −dp/dx = 3/(Re ly²) for U_b = 1 (0.75/Re for `ly` = 2), against 12/(Re ly²) with two no-slip walls.
* **Output:** in `log.dat`, τ_w and Re_τ are those of the no-slip wall.
* **Scalars:** they keep their own wall conditions (`sbc_bot`, `sbc_top`).
* **Check:** a run with the free-slip wall at y = `ly` is the exact mirror image (to round-off) of the same run
  with the free-slip wall at y = 0, when gravity and the scalar wall conditions are mirrored too.

**Wall-normal domain and grid.**
* `ly` in `&grid` is the box height: the walls are at y = 0 and y = `ly`, default 2. `re`, `lx` and `ly`
  use the same length unit. With `ly = 2` and `re` based on the half-height, the set-up is the classical channel
  with half-height 1, as in HiCHAN (there with walls at ±1).
* Restart files store the wall positions. A restart is accepted if the box height is the same, so files of
  earlier versions (walls at −1 and 1) can be restarted with `ly = 2`.
* `grid_type = 'auto'` (default) chooses:
  * `'tanh'` (refined at both walls) with two no-slip walls;
  * `'tanh_top'` (refined only at y = `ly`) with `bc_bot = 'freeslip'`;
  * `'tanh_bot'` (refined only at y = 0, the mirror image) with `bc_top = 'freeslip'`.
* With the same `grid_alpha`, `'tanh_top'`/`'tanh_bot'` with ny points is exactly one half of the `'tanh'` grid of a
  channel twice as deep with 2·ny − 1 points. So an open channel of depth h gets the near-wall resolution of a
  closed channel of half-height h.
* Other choices: `'tanh'` or `'uniform'` explicitly, or `'file'` (ny values of y from the bottom to the top
  wall).
* A coarse grid near the free-slip wall suits shear-driven flow. Flow driven at the free-slip wall needs
  resolution there too. Example: in Rayleigh–Bénard onset with ny = 33, `'tanh_top'` gives Ra_c to 6e-4 and
  `'tanh'` to 5e-5; at ny = 65, `'tanh_top'` reaches 7e-5.
* `tools/os_mode.py` (TS-wave initial mode) assumes `ly = 2` (lengths in units of the half-height).

The scalars obey `∂s/∂t + u·∇s = (1/(Re Pr)) ∇²s + Q`, and act on the flow as `−Σ ri(n) s_n e_g`. Here `e_g`
is the direction of gravity. `ri > 0` is a scalar that makes the fluid lighter (temperature), `ri < 0` one
that makes it heavier (salinity).

The discretisation is the one of HiCHAN's `'vv'` option:

* All y-operators are A2-premultiplied compact operators, `[(1+c kx²)A2 − cB2] f = A2 r`.
* Dirichlet conditions use the wall-row (τ) correction.
* The two conditions v = dv/dy = 0 are imposed by superposition with two homogeneous solutions (φ = 1 at
  one wall). Their 2×2 matrix depends only on kx and the RK substep, and is recomputed only when Δt changes.
* Everything is banded and O(Ny) per wavenumber. One GPU thread handles one kx and marches in y; no
  iterations are needed.
* Continuity holds to round-off at all interior points. The wall condition dv/dy = 0 uses a 6-point
  one-sided derivative (`vv_neumann = 'explicit'`) or the compact boundary row (`'compact'`). With
  `'explicit'`, `div_max` in `log.dat` shows the truncation-size wall-row value.

## Melting ice

`&melt_model` with `melt = .true.` turns the wall y = 0 into a melting ice interface. The full derivation
and a comparison with Wells & Worster (2011) are in `docs/three_equation_melt.tex` (PDF next to it). Heat
conduction into the ice is neglected.

**The conditions**, at every x and in the units of the code:

```
theta_b = Lambda s_b + theta_f                 liquidus
m*      = St kappa_T dtheta/dy|_0              heat:  latent heat = conductive heat flux from the water
kappa_S ds/dy|_0 = m* (s_b + sigma)            salt:  diffusive flux = dilution by the meltwater
v(x,0)  = m*(x),  u(x,0) = 0                   meltwater velocity (melt_blowing; interface fixed at y = 0)
```

* m* = ρ_i m/(ρ_w U) is the melt rate as a water velocity, and κ = 1/(Re Pr).
* θ = scalar `melt_t` and s = scalar `melt_s`. Both need `sbc_bot = 'D'`; their `sval_bot` are replaced
  by θ_b(x) and s_b(x).
* `melt_s = 0` gives the Stefan condition (θ_b = θ_f).
* `examples/ice_layer/input.nml` shows how to obtain St, Λ, θ_f and σ for sea water.

**Implicit coupling.** The interface values are the Dirichlet values of the scalar solves of each RK
substep. The conditions must hold with the wall gradients of the *new* scalars.
* The theta-method solve is linear in the wall value. For every Fourier mode k, the new wall gradient
  is g0_k + R_k w_k, where R_k < 0 is the response to a unit wall value. R_k is computed with the solver
  itself and cached per substep and Δt.
* Each iteration solves all scalars with the current θ_b(x) and s_b(x). It then re-solves the local
  conditions at every x, where they reduce to a quadratic for the interface salinity with exactly one
  physical root.
* The constant R_ref = (min R_k + max R_k)/2 replaces R_k in that local step, so the iteration converges
  for every mode whatever Δt. At convergence (`melt_tol`) the conditions hold exactly.
* In the tests it took 1–7 iterations per substep, logged in `log.dat`. Each costs one batched scalar solve
  and five x-FFTs.
* The scalars are solved before the velocity in each substep. The velocity then uses m* as its wall value
  at y = 0.

**Mean outflow.** In 2D the mean meltwater velocity is the same at every y (continuity). So it leaves the
domain through y = `ly`, where the mean wall-normal velocity equals the mean of m*. With `melt_blowing =
.false.` the ice wall stays impermeable.

**Time stepping.** Crank–Nicolson (`scal_theta = 0.5`) rings after abrupt changes at the interface, e.g.
from an initial state that does not satisfy the melt condition; m* then oscillates from step to step and the
oscillation decays slowly on fine grids. `scal_theta = 1` (implicit Euler, first order in time) damps it, and
can be used to start a run (then restart with 0.5).

**Not modelled:**
* the motion of the interface (it recedes slowly and is kept fixed at y = 0);
* heat conduction into the ice;
* a nonlinear equation of state (the density maximum of fresh water near 4 °C);
* the pressure dependence of the freezing point.

## Contents

```
src/types.f90      kinds
src/param.f90      input (namelists in input.nml)
src/compact.f90    grid, compact-scheme coefficients on non-uniform grids, quadrature (from HiCHAN)
src/decomp.f90     1D (slab) decomposition, x-lines <-> y-lines transpose (MPI_Alltoallv, CUDA-aware)
src/fft.f90        x-transforms (cuFFT / FFTW), dealiasing
src/ysolve.f90     wall-normal column solvers: derivatives, scalar Helmholtz, velocity-vorticity step
src/melt.f90       three-equation melt condition at an ice wall, coupled implicitly to the scalar solves
src/rk.f90         nonlinear terms (rotational or advective), buoyancy, RK3 step, time-step limit
src/fields.f90     field storage
src/stats.f90      statistics (Parseval, spectral space), log.dat
src/io.f90         restart files (MPI-IO), 2D fields (MPI-IO + XDMF)
src/initflow.f90   initial conditions
src/timer.f90      optional timers
src/main.f90       main program
configs/           arrhenius.mk, generic_gpu.mk
examples/          input_reference.nml (every input item), ts_wave, rayleigh_benard (two active scalars),
                   rayleigh_benard_freeslip (free-slip bottom wall), ice_layer (template: water under
                   melting ice: three-equation condition at y = 0, free-slip at y = ly, gravity +y),
                   salt_fingers (two scalars with opposite buoyancy), salt_fingers_gpu (production-size
                   template for one Arrhenius node), arrhenius_job.sh
tools/             os_mode.py (Orr–Sommerfeld mode for init_type = 'mode'; needs numpy, scipy),
                   plot_field.py (plots the 2D field output, Scientific Colour Maps, colour ranges;
                   needs numpy, matplotlib)
```

## Building

**CPU** (any MPI + FFTW3):

```bash
make
```

**NSC Arrhenius** (GH200, NVIDIA HPC SDK):

```bash
module load GPU/buildenv-nvhpc/25.9-cu13.0
```

```bash
make CONFIG=configs/arrhenius.mk
```

This builds with `-acc -cuda -gpu=cc90`, cuFFT and `-D_CUDA`. The transposes call `MPI_Alltoallv` on device
buffers, so the MPI must be CUDA-aware (the HPC SDK's MPI is). On one node the traffic stays on NVLink.

**Other builds:** `make DEBUG=1` adds run-time checks. `make ACC_HOST=1` builds with `gfortran -fopenacc`
(OpenACC host fallback, FFTW instead of cuFFT) to check the directives without a GPU.

**Running:** one MPI rank per GPU (`examples/arrhenius_job.sh`: 1 node, 4 tasks, 4 GPUs). The code prints the
GPU used by each rank and stops if two ranks share a GPU. The number of ranks must not exceed nx/2 or ny.

**Note:** this code has not yet been compiled with nvfortran or run on a GPU: no NVIDIA compiler or GPU was
available where it was written. Everything else was tested on CPU, including the OpenACC code path via the
`ACC_HOST=1` build (see below). Build it on Arrhenius and run `examples/ts_wave` and
`examples/rayleigh_benard` first: they take seconds and check the GPU results against known values.

### GPU performance in 2D

The FFTs, transposes and pointwise products are well parallel. The wall-normal solves are not: they use
one GPU thread per wavenumber, i.e. only nx/2/(number of GPUs) threads per GPU (512 for nx = 4096 on 4 GPUs),
each marching sequentially over ny points. In 3D HiCHAN has nx·nz/2 such columns; in 2D there are far fewer.
Such kernels are latency bound, and their run time hardly depends on the number of threads.

The scalar Helmholtz solves are therefore **batched**: one kernel over all (kx, scalar) pairs, with
nscal·nkx threads and one work column per pair. With two scalars this costs about one scalar solve instead
of two. The threads of a warp belong to the same scalar, so Dirichlet and Neumann scalars do not diverge
within a warp (if nkx is a multiple of 32). The results are bit-identical to solving the scalars one by one.

A further step, not implemented: within an RK substep the scalar solves are independent of the velocity
step, so the two could run concurrently on separate OpenACC queues (each needs its own work arrays).

GPUs pay off for large nx. For small or moderate grids, one GPU, or the CPU build, can be as fast as four.
Use `timing = .true.` to see where the time goes.

## Input

`examples/input_reference.nml` lists every item with its meaning and default. The namelists are:

| namelist | contents |
|---|---|
| `&grid` | `nx` (Fourier modes, even), `ny` (points incl. walls), `lx`, `ly` (box height; walls at y = 0 and ly; default 2), `grid_type` (`auto`/`tanh`/`tanh_top`/`tanh_bot`/`uniform`/`file`), `grid_alpha`, `grid_file`, `fd_order` (4 or 6), `dealias` |
| `&physics` | `re`, `forcing` (`cfr`/`cpg`), `ubulk`, `dpdx`, `vv_neumann` (`explicit`/`compact`), `nl_form` (`rotational`/`advective`) |
| `&time` | `nstep`, `tmax`, `wall_time_max`, `fixed_dt`, `dt`, `dt_max`, `dt_growth`, `cfl`, `cfl_visc` |
| `&init` | `init_type` (`laminar`/`noise`/`mode`/`restart`), `init_amp`, `init_seed`, `mode_file`, `mode_kx`, `restart_file`, `reset_time` |
| `&output` | `data_dir`, `iout_log`, `iout_stats`, `iout_save_stats`, `iout_restart`, `iout_field`, `tstats_start`, `mode_log_kx`, `timing` |
| `&scalars` | `nscal` (default 2); per scalar `pr(n)`, `scal_theta(n)`, `sbc_bot(n)`/`sbc_top(n)` (`D`/`N`), `sval_bot(n)`/`sval_top(n)`, `ssrc_const(n)`, `ssrc_u(n)`, `sinit(n)` (`linear`/`zero`/`one`/`noise`); `scal_form` (`advective`/`divergence`/`skew`) |
| `&buoyancy` | `gravity` (x, y direction), `ri(n)` (Richardson number per scalar, 0 = passive) |
| `&melt_model` | `melt`, `melt_t`, `melt_s` (temperature and salinity scalars; `melt_s = 0`: Stefan), `melt_st` (St), `melt_lambda` (Λ), `melt_thf` (θ_f), `melt_sigma` (σ), `melt_blowing`, `melt_maxit`, `melt_tol` |

The adaptive time step obeys the advective CFL limit, the explicit-diffusion limit (scalars with θ < ½), and
the internal-gravity-wave limit `Δt ≤ cfl/N`, with `N² = |ri| max|dS/dy|`. It is kept piecewise constant,
because the implicit matrices depend on Δt. Between two walls at distance h, the Rayleigh number of a scalar
with wall difference Δs is `Ra = ri Δs ly³ re² pr` (`8 ri Δs re² pr` for `ly = 2`).

## Output (in `data_dir`)

* **`log.dat`** columns:
  1. step
  2. t
  3. Δt
  4. bulk velocity
  5. applied −dp/dx
  6. τ_w
  7. Re_τ = u_τ δ/ν, with δ = half the wall distance, or the full wall distance (open-channel depth) with a
     free-slip wall
  8. max |div u|
  9. fluctuation energy
  10. energy of mode `mode_log_kx`
  11. onwards: dS/dy at the bottom and top wall, for each scalar
  12. with melting, after the scalars: x-averages of the melt rate m*, θ_b and s_b, the number of coupling
      iterations in the last substep, and the last change of the interface values
* **`melt_NNNNNNN.dat`** (with melting, at every field output): x, m*, θ_b, s_b along the interface.
* **`stats.dat`**: time-averaged profiles U, V, ⟨u′u′⟩, ⟨v′v′⟩, ⟨u′v′⟩, and per scalar S, ⟨s′s′⟩, ⟨u′s′⟩,
  ⟨v′s′⟩.
* **`restart.bin` / `restart.hdr`**: u, v and the scalars as global (nx/2, ny) complex arrays, independent of
  the number of ranks. φ is rebuilt from v at a restart; its wall values do not affect the solution.
* **`fld_NNNNNNN_<u|v|s1|s2>.bin` + `.xmf`**: physical fields on the dealiased mx × ny grid, readable with
  ParaView/VisIt or `tools/plot_field.py`:

```bash
python3 tools/plot_field.py data 8000 s1 s2 v -o fields.png
```

Options of `plot_field.py` (full list with `-h`):

* **Colour maps:**
  * `--scmap NAME` sets the colour map of the scalars; `--cmap NAME` sets it for all fields.
  * `NAME` is a Scientific Colour Map (Crameri: acton, batlow, vik, roma, …), read from
    `DIR/NAME/NAME.txt` (or `.mat`/`.cpt`); a file path also works.
  * `DIR` is `--cmap-dir`, else `$SCM_DIR`, else `/scratch/geert/matlab_colours`.
  * A matplotlib name also works, and `NAME_r` reverses a map.
* **Colour ranges:**
  * `--srange MIN MAX` sets the range of all scalars.
  * `--clim NAME MIN MAX` sets the range of one field; it can be repeated and overrides `--srange`.
  * Values outside the range are saturated, with arrows on the colour bar.
  * Without these options, scalars use the range of their wall values.

```bash
python3 tools/plot_field.py data 8000 s1 s2 v --scmap acton --cmap vik --srange 0 1 --clim s2 0 0.3
```

## Validation (CPU, in this repository)

| test | result |
|---|---|
| Against HiCHAN `'vv'` with nz = 2 (one spanwise mode), from the same state: nonlinear, 200 steps, rotational form, CFR, two active scalars, oblique gravity (−0.6, −0.8), one Neumann wall, a scalar source | u, v, s1, s2 identical to round-off: max difference 6e-15 (fields of order 1) |
| TS wave, Re = 5000, α = 1.12, ny = 65 (`examples/ts_wave`) | decay rate −2.78912e-3 vs Orr–Sommerfeld −2.78290e-3: error 6.2e-6, as for HiCHAN `'vv'` (6.4e-6). HiCHAN `'vv'` on the same input: −2.78930e-3; the small difference comes from the different initial u. The fitted late-time rates agree to 2e-8 |
| Rayleigh–Bénard onset driven by both scalars (Ri split 70/30, `examples/rayleigh_benard`) | Ra_c = 1707.80 (linear theory 1707.76; HiCHAN `'vv'`: 1707.80) |
| Decomposition independence, 1, 2, 3, 4 ranks (incl. uneven splits) | restart files and `log.dat` bit-identical |
| Restart: 20 steps vs 10 + restart + 10 (2 → 3 ranks, adaptive Δt) | identical to round-off (5e-15); `log.dat` identical |
| OpenACC code path (`ACC_HOST=1`, 4 ranks) | same as the CPU build to 7e-15, `log.dat` identical |
| `DEBUG=1` (bounds checks, FP traps), nonlinear two-scalar case on 3 ranks | clean |
| Batched scalar solve (both scalars in one kernel; one Dirichlet, one Neumann/Dirichlet with source) | bit-identical to the per-scalar solves |
| Free-slip bottom, laminar CFR (`bc_bot = 'freeslip'`) | open-channel profile kept exactly, −dp/dx = 0.75/Re, τ_w = 1.5/Re (top wall) |
| Walls at 0 and ly = 1, laminar, CFR and CPG, no-slip (`'tanh'`) and free-slip (`'auto'` → `'tanh_top'`) | exact: −dp/dx = 0.012 = 3/(Re (h/2)²) and 0.003 = 3/(Re h²); U_b = 1 at those gradients; Re_τ = 38.73 and 54.77 as expected |
| Rayleigh–Bénard onset with walls at 0 and 1, two active scalars | no-slip: Ra_c = 1707.80; free-slip bottom: 1100.59 (`'tanh'`, ny = 33), 1100.57 (`'tanh_top'`, ny = 65), identical to `ly = 2` |
| y from 0 to `ly` always (default `ly = 2`) instead of −1..1 | same results as the −1..1 version to round-off (2e-14); restart files of that version accepted; TS wave, Rayleigh–Bénard (no-slip and free-slip) and exact laminar checks unchanged; 1 vs 4 ranks bit-identical |
| Default domain and grid after these changes | bit-identical to the previous version; restart headers of the previous version are still read |
| ly = 2.5, free-slip, `'auto'` grid, nonlinear, two active scalars | bit-identical on 1 and 4 ranks; 150 steps + restart (2 → 3 ranks) + 150 steps agrees with 300 steps to 1.5e-13; OpenACC host build agrees to 7e-13; `DEBUG=1` clean |
| Rayleigh–Bénard onset, free-slip bottom / no-slip top, two active scalars (`examples/rayleigh_benard_freeslip`) | Ra_c = 1100.59 (ny = 33), 1100.64 (ny = 65); linear theory 1100.65 (rigid–free) |
| Free-slip wall at y = `ly` (`bc_top`): no-slip and free-slip-at-y = 0 runs after the change | bit-identical to the previous version |
| Free-slip at y = `ly`, laminar, `ly` = 1, CFR and CPG (`'auto'` → `'tanh_bot'`) | exact: −dp/dx = 3/Re, U_b = 1, Re_τ = 54.77; grid the mirror image of `'tanh_top'` |
| Mirror test: nonlinear two-scalar run (`ly` = 2.5, oblique gravity, Neumann scalar wall, source) with the free-slip wall at y = 0, against the mirrored set-up with the free-slip wall at y = `ly` (gravity, wall values and Neumann gradients mirrored), 200 steps, 2 vs 3 ranks | mirror images to round-off (≤ 3e-14); identical −dp/dx and τ_w |
| Rayleigh–Bénard onset, free-slip wall at y = `ly` | Ra_c = 1100.59 (as with the free-slip wall at y = 0; theory 1100.65) |
| Free-slip, nonlinear two-scalar case | bit-identical on 1 and 4 ranks; OpenACC host build agrees to 8e-14; ∂u/∂y at the free-slip wall converges to 0 with refinement (0.25, 0.019, 0.0017 for ny = 49, 97, 193, vs ~6.3 at the no-slip wall) |
| Melting, steady Stefan problem (no salinity, St = 0.5, ice at y = 0, θ = 1 at y = 1) | m* = κ ln(1 + St) to 6e-9; temperature profile (e^{Vy/κ} − 1)/(e^{V/κ} − 1) to 3e-10; v = m* uniform |
| Melting, steady three-equation problem (St = 0.5, Λ = −0.5, κ_T = 1, κ_S = 0.5) | m*, θ_b, s_b to 1.4e-8, 3.9e-8, 3.9e-8 (ny = 65) and 1.6e-9, 4.6e-9 (ny = 129), against the exact solution (exponential profiles, root of one equation); 1–2 coupling iterations per substep at steady state |
| Melting, convection under ice (sea-water-like Λ, θ_f, St; Pr = 7, Sc = 50; 128 × 97) | stable, 3–4 coupling iterations per substep; Crank–Nicolson and θ = 1 agree; bit-identical on 1 and 3 ranks; restart (2 → 4 ranks) to 7e-14; OpenACC host build to 6e-14; `DEBUG=1` clean |
| Runs without melting after the change | bit-identical |
| Salt fingers, R_ρ = 1.2, Pr = 7, Sc = 21 (`examples/salt_fingers`, 256 × 129) | fingers grow from noise and saturate; wall flux of salt ×3.3 and of heat ×2.3 the conductive value; divergence ~1e-5 (wall rows) |

The TS-wave and Rayleigh–Bénard checks:

```bash
cd examples/ts_wave
python3 ../../tools/os_mode.py --re 5000 --alpha 1.12 --ny 65 --grid tanh --galpha 4.0 -o mode.in
mpirun -np 1 ../../build/solver2d
python3 check.py
```

```bash
cd examples/rayleigh_benard
python3 check.py ../../build/solver2d "mpirun -np 1"
```

## Relation to HiCHAN

These parts are taken over unchanged or with z removed:

* the compact schemes and grid (`compact.f90`, unchanged)
* the banded column solvers, the scalar θ-method solve with its τ/influence-matrix wall conditions, and
  the `'vv'` velocity–vorticity step with CFR correction (`ysolve.f90`)
* the RK3 coefficients, nonlinear forms, buoyancy, time-step limits, statistics, restart/field I/O,
  initial conditions, input namelists, the GPU device binding, and the OpenACC/cuFFT stream design

These parts differ:

* There is no ω_y equation and no w.
* The decomposition is a 1D slab with one `MPI_Alltoallv` transpose. cuDecomp, diezDecomp and process grids
  are not needed in 2D.
* The primitive-variable `'im'` method of HiCHAN is not included; only the velocity–vorticity method is.
