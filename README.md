# 2D solver — DNS of two-dimensional channel flow with active scalars (CPU and Grace Hopper GPUs)

`solver2d` solves the incompressible Navier–Stokes equations in a two-dimensional plane channel
(x periodic, walls at y = ±1), together with active scalars (two by default, up to 16) that act on the flow
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
  any direction in the x–y plane.
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

## Contents

```
src/types.f90      kinds
src/param.f90      input (namelists in input.nml)
src/compact.f90    grid, compact-scheme coefficients on non-uniform grids, quadrature (from HiCHAN)
src/decomp.f90     1D (slab) decomposition, x-lines <-> y-lines transpose (MPI_Alltoallv, CUDA-aware)
src/fft.f90        x-transforms (cuFFT / FFTW), dealiasing
src/ysolve.f90     wall-normal column solvers: derivatives, scalar Helmholtz, velocity-vorticity step
src/rk.f90         nonlinear terms (rotational or advective), buoyancy, RK3 step, time-step limit
src/fields.f90     field storage
src/stats.f90      statistics (Parseval, spectral space), log.dat
src/io.f90         restart files (MPI-IO), 2D fields (MPI-IO + XDMF)
src/initflow.f90   initial conditions
src/timer.f90      optional timers
src/main.f90       main program
configs/           arrhenius.mk, generic_gpu.mk
examples/          input_reference.nml (every input item), ts_wave, rayleigh_benard (two active scalars),
                   salt_fingers (two scalars with opposite buoyancy), salt_fingers_gpu (production-size
                   template for one Arrhenius node), arrhenius_job.sh
tools/             os_mode.py (Orr–Sommerfeld mode for init_type = 'mode'; needs numpy, scipy),
                   plot_field.py (plots the 2D field output; needs numpy, matplotlib)
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
GPUs therefore pay off for large nx. For small or moderate grids, one GPU, or the CPU build, can be as fast
or faster than four GPUs. Use `timing = .true.` to see where the time goes. If the column solves dominate,
the next step would be to batch the scalar solves into one kernel, which needs per-scalar work arrays
(not done yet).

## Input

`examples/input_reference.nml` lists every item with its meaning and default. The namelists are:

| namelist | contents |
|---|---|
| `&grid` | `nx` (Fourier modes, even), `ny` (points incl. walls), `lx`, `grid_type` (`tanh`/`uniform`/`file`), `grid_alpha`, `grid_file`, `fd_order` (4 or 6), `dealias` |
| `&physics` | `re`, `forcing` (`cfr`/`cpg`), `ubulk`, `dpdx`, `vv_neumann` (`explicit`/`compact`), `nl_form` (`rotational`/`advective`) |
| `&time` | `nstep`, `tmax`, `wall_time_max`, `fixed_dt`, `dt`, `dt_max`, `dt_growth`, `cfl`, `cfl_visc` |
| `&init` | `init_type` (`laminar`/`noise`/`mode`/`restart`), `init_amp`, `init_seed`, `mode_file`, `mode_kx`, `restart_file`, `reset_time` |
| `&output` | `data_dir`, `iout_log`, `iout_stats`, `iout_save_stats`, `iout_restart`, `iout_field`, `tstats_start`, `mode_log_kx`, `timing` |
| `&scalars` | `nscal` (default 2); per scalar `pr(n)`, `scal_theta(n)`, `sbc_bot(n)`/`sbc_top(n)` (`D`/`N`), `sval_bot(n)`/`sval_top(n)`, `ssrc_const(n)`, `ssrc_u(n)`, `sinit(n)`; `scal_form` (`advective`/`divergence`/`skew`) |
| `&buoyancy` | `gravity` (x, y direction), `ri(n)` (Richardson number per scalar, 0 = passive) |

The adaptive time step obeys the advective CFL limit, the explicit-diffusion limit (scalars with θ < ½), and
the internal-gravity-wave limit `Δt ≤ cfl/N`, with `N² = |ri| max|dS/dy|`. It is kept piecewise constant,
because the implicit matrices depend on Δt. Between two walls (gap 2h) the Rayleigh number of a scalar is
`Ra = 8 ri Δs re² pr`.

## Output (in `data_dir`)

* **`log.dat`** columns:
  1. step
  2. t
  3. Δt
  4. bulk velocity
  5. applied −dp/dx
  6. τ_w
  7. Re_τ
  8. max |div u|
  9. fluctuation energy
  10. energy of mode `mode_log_kx`
  11. onwards: dS/dy at the bottom and top wall, for each scalar
* **`stats.dat`**: time-averaged profiles U, V, ⟨u′u′⟩, ⟨v′v′⟩, ⟨u′v′⟩, and per scalar S, ⟨s′s′⟩, ⟨u′s′⟩,
  ⟨v′s′⟩.
* **`restart.bin` / `restart.hdr`**: u, v and the scalars as global (nx/2, ny) complex arrays, independent of
  the number of ranks. φ is rebuilt from v at a restart; its wall values do not affect the solution.
* **`fld_NNNNNNN_<u|v|s1|s2>.bin` + `.xmf`**: physical fields on the dealiased mx × ny grid, readable with
  ParaView/VisIt or `tools/plot_field.py`:

```bash
python3 tools/plot_field.py data 8000 s1 s2 v -o fields.png
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
