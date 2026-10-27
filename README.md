# nekabl

Exploratory analysis of **atmospheric boundary layer (ABL) LES** run with
[Neko](https://github.com/ExtremeFLOW/neko) and Nek5000. `z` is always the
vertical direction, `x`–`y` are the homogeneous directions (Nek5000 runs with
`y` up are rotated on reading). Every formula and assumption is listed in
**[METHODS.md](METHODS.md)**.

The package is a set of short, independent scripts with the same structure:
*compute functions* that take and return `xarray` objects, and *plot functions*
that return `fig, ax(s)` and accept `ax=` so that panels can be combined.
It is meant to be read and extended, not configured.

```
nekabl/
├── case.py           read the Neko .case file -> g, θ0, z0, surface flux, f, Ug, LES model, stats settings
├── logfile.py        parse the run log -> dt, CFL, Krylov iterations, wall time per step, warnings
├── stats.py          read statistics: CSV (basic/full, SGS), 3D stats fld files, Nek5000 avg/rms/rm2
├── fields.py         read field files (fld) on box meshes -> (time, z, y, x); planes, averages, uniform grids
├── sem.py            GLL helpers: exact d/dz, integrals and interpolation on the spectral-element grid
├── profiles.py       raw stats -> mean profiles (central moments, fluxes, totals, Ri) + profile plots
├── diagnostics.py    u*, θ*, L, zi (5 definitions), w*, MOST / wall-model / SGS / resolution / time-step checks
├── turbulence.py     TKE, <w'θ'> and <θ'²> budgets; K_m, K_h, Pr_t; spectra; autocorrelation; quadrants; PDFs
├── visualization.py  horizontal and vertical slices
├── most.py           Monin–Obukhov functions (identical to the Neko MOST wall model)
├── plotstyle.py      ONE place for the figure look (font size, DPI, LaTeX, colours, line styles)
├── run.py            Run: one object per simulation folder, Neko or Nek5000 (stats, profiles, fields, log)
├── nek5000.py        Nek5000 interface: y-up rotation, KTH statistics (sts files), .par/SIZE
├── summary.py        one-page text summary of a run            (command: nekabl-summary)
└── explore.py        all figures + summary for one or more runs  (command: nekabl-explore)
examples/
└── main.py           full walkthrough on the example case (cell-by-cell or as a script)
tests/
└── check_reader_vs_pymech.py   the fld reader returns exactly pymech's data
METHODS.md            what every function computes, formulas, assumptions
```

## Install

```bash
cd nekabl
pip install -e .            # dependencies: numpy, xarray, pandas, matplotlib
```

No compiled code, no MPI, no pymech/pysemtools needed: the fld reader is ~100
lines of numpy.

## Quick start

```python
import nekabl as nk

run  = nk.Run("path/to/case", label="neutral")      # Neko or Nek5000, detected
prof = run.profile(2000, 3000)                       # mean profiles, u*, L, zi... in prof.attrs
fld  = run.fields(tmin=2000)                         # field snapshots

# compare simulations: every plot takes {label: data}; colour = simulation
runs = [run, nk.Run("path/to/other", label="fine grid")]
nk.plot_profiles({r.label: r.profile() for r in runs}, ["M", "T"], zi=True)   # z / own zi
nk.plot_fluxes({r.label: r.profile() for r in runs}, ["uw", "wt"], zi=True)
```

or with the plain functions:

```python
case = nk.read_case("path/to/case")                 # finds the .case file
raw  = nk.load_stats("path/to/case/output", case=case)   # all *stats*.csv -> raw moments (time, z)

prof = nk.derive(nk.time_average(raw, 2000, 3000), case=case)   # mean profiles (z)
sc   = nk.scales(prof, case)                         # u*, wθ_s, θ*, L, zi, w*, -zi/L ...

fig, axs = nk.plot_profiles(prof, ["M", "T", "tke"], zmax=1500)
fig, axs = nk.plot_fluxes(prof, ["uw", "vw", "wt"], zi=sc["zi"])
fig, ax  = nk.plot_budget(nk.tke_budget(prof), zi=sc["zi"])

fld = nk.read_fields("path/to/case/output", rename={"s1": "e", "s3": "nut"}, tmin=2000)
fig, ax  = nk.plot_hslice(fld, "w", z=0.5 * sc["zi"])
fig, ax  = nk.plot_spectra(nk.spectrum(fld, "w", [100, 300], "radial"), premultiplied=True, zi=sc["zi"])
```

## From the command line (e.g. at the end of a job script)

```bash
nekabl-summary  path/to/case                    # text summary (case, log, files, u*, L, zi ...)
nekabl-explore  path/to/case                    # ~30 figures + summary.txt in ./nekabl_figures
nekabl-explore  runA runB --labels coarse fine --t0 2000 --out figs --format pdf
nekabl-explore  stable_16_TKE --t0 28799 --out figs                  # Nek5000: parameters read from .par/.usr
nekabl-explore  --config runs.json              # per-run options (see nekabl/explore.py)
nekabl-explore  path/to/case --t0 5000 --evolution             # + thin profile of every stats window
nekabl-explore  path/to/case --t0 8000 --evolution 5000 10000  # mean from 8000 s, thin lines 5000-10000 s
```

`--evolution` (one case only) draws, behind each mean profile (black), the
profile of every statistics window as a thin line coloured by time (viridis:
yellow = first window, violet = last; colour bar on the right), so a trend
is visible at a glance. Without values it uses the averaging window, with two
values from T0 to T1. Profile figures are titled with the averaging window
and the output interval of the statistics.
From Python: `nk.plot_overview(prof, "momentum", zi=True, evolution=run.evolution(5000, 10000))`
(also `plot_profiles`, `plot_fluxes`, `plot_hodograph`, `plot_sgs`,
`plot_eddy_viscosity`; `plot_similarity(..., evolution=nk.similarity_evolution(evo, case))`).

Each figure is made independently: anything that cannot be computed (basic
statistics, no field files, no log ...) is skipped and the reason is written
in `summary.txt`. Field-file figures use only the snapshots inside the
averaging window (the last one if there is none). With *basic* statistics, for example, the TKE budget has no
turbulent/pressure transport and the summary says so.

The walkthrough `examples/main.py` runs everything on the example case and
writes ~25 figures in `examples/figures/`.

## Entrainment and passive scalars

```python
run = nk.Run("path/to/case")                       # passive scalars with statistics files are found automatically
ent = run.entrainment(t0, t1)                      # all scalars + U, V; per statistics window
ent = run.entrainment(t0, t1, scalars=["co2"])     # only some (names or symbols)
print(nk.entrainment_table(ent))                   # zi, EZ, jumps, F_e, w_e, A, E, SGS share, Ri_w
nk.plot_inversion(run.profile(t0, t1), scalars=["t", "co2"])   # jump construction + flux with the EZ
nk.plot_entrainment(ent)                           # time series: zi/EZ, w_e, A, SGS share, E vs Ri_w
ent, bud = run.entrainment(t0, t1, budget=True)    # growth-based w_e and mixed-layer-budget F_e
nk.plot_entrainment_budget(bud)
fld = run.fields(tmin=t0); h = nk.inversion_height_map(fld, zi)          # local inversion height
comp = nk.interface_composite(fld, h, scalars=["co2"]); nk.plot_interface_composite(comp)
```

`nekabl-explore ... --scalars co2 temperature` restricts the scalar and entrainment figures (default: all).
Figures `18_*` (profiles, time series, budget) and `19_*` (inversion map, interface composites) are added;
the numbers go to `summary.txt`. Definitions and limits: `METHODS.md`, section 7. `tests/check_entrainment.py`
checks everything against an analytic jump model.

## Statistics or field files?

Field files are big and give few samples; the CSV statistics are tiny and are
averaged over every `compute_value`. **Use the CSV statistics for anything
averaged**, and the field files only for what needs instantaneous data.

| What                                                        | Source                                   | Function |
|-------------------------------------------------------------|------------------------------------------|----------|
| mean wind, θ, wind direction, hodograph                     | `fluid_stats` + `scalar_stats` (basic)   | `derive`, `plot_profiles`, `plot_hodograph` |
| variances, resolved fluxes, resolved TKE                    | basic sets                               | `derive`, `plot_fluxes` |
| SGS stresses / fluxes, ⟨νt⟩, ⟨αt⟩, total fluxes, SGS Pr     | `fluid_sgs_stats`, `scalar_sgs_stats`    | `derive`, `plot_sgs` |
| u*, θ*, L, zi, w*, −zi/L, time series of them               | basic + SGS sets                         | `scales`, `timeseries` |
| MOST similarity, wall-model check                           | basic + SGS sets + case file             | `similarity`, `wall_model_check` |
| surface heat flux / temperature: LES vs wall model          | basic + SGS sets + case file (+ log)     | `surface_check`, `plot_surface`, `Run.wall_model_series` |
| cross-isobaric transport, wind turning, Ekman balance       | basic + SGS sets + u_g, f                | `ekman_transport`, `plot_ekman`, `Q_cross` in `derive` |
| all momentum / heat / scalar / stability profiles at once   | whatever is available                    | `plot_overview(prof, "momentum" \| "heat" \| "scalar" \| "stability")` |
| Richardson numbers (gradient, bulk, flux)                   | basic (+ SGS sets for Ri_f total)        | `derive` → `Ri_g`, `Ri_b`, `Ri_f`; `plot_overview(..., "stability")` |
| K_m, K_h, Pr_t, mixing length                               | basic + SGS sets                         | `eddy_viscosity` |
| TKE budget: shear and buoyancy production (resolved + SGS)  | basic + SGS sets                         | `tke_budget` |
| TKE budget: turbulent and pressure transport                | **full** fluid set                       | `tke_budget` |
| TKE budget: dissipation                                     | field files (Deardorff `e`, or νt + gradients) | `dissipation_from_fields` → `tke_budget(eps=)` |
| skewness / kurtosis (Su, Sw, Sθ ...)                        | **full** sets                            | `derive` |
| ⟨w'θ'⟩ and ⟨θ'²⟩ budgets                                     | **full** scalar set (+ basic fluid)      | `heat_flux_budget`, `theta_variance_budget` |
| slices (horizontal / vertical)                              | field files                              | `plot_hslice`, `plot_hslices`, `plot_vslice` |
| spectra, cospectra, autocorrelation, integral scales        | field files                              | `spectrum`, `autocorrelation`, `integral_length` |
| quadrant analysis, PDFs, joint PDFs                         | field files                              | `quadrant_analysis`, `pdf`, `plot_joint_pdf` |
| SGS TKE (Deardorff `e`) → resolved-TKE fraction (Pope 80 %) | field files (or a `scalar_stats` on TKE) | `from_fields(..., extra=["e"])`, `resolution` |
| any other mean you forgot to put in the stats               | field files                              | `from_fields` |
| largest stable dt of a snapshot                             | field files                              | `cfl_dt` |
| dt, CFL, iterations, cost                                   | log                                      | `read_log`, `plot_timestep`, `summarize` |

`from_fields` returns exactly the same *raw* Dataset as the CSV route, so
`time_average`/`derive`/all plots work on it too (handy to compare the two).

## Conventions (read once)

* **Raw vs derived.** `load_stats` returns Neko's raw moments ⟨u⟩, ⟨uw⟩,
  ⟨uuw⟩, ⟨p ∂u/∂z⟩ … with dims `(time, z)`. `derive` turns them into physical
  quantities. In the derived Dataset the same letters mean *central* moments
  (`uw` = ⟨u'w'⟩, `wtt` = ⟨w'θ'θ'⟩), upper case are means (`U, V, W, P, T`).
  Central moments of any order are computed by inclusion–exclusion, so they are
  exact.
* **Time averaging.** Each CSV time stamp is the average over the window since
  the previous write (the first starts at the stats `start_time`). The window
  lengths are kept as the coordinate `window`, and `time_average` weights by
  them. Raw moments are averaged *before* computing central moments.
* **Scalar names.** The Neko `s` is replaced by a symbol, `t` for temperature by
  default: `load_stats(..., scalars={"temperature": "t", "humidity": "q"})`.
* **SGS sign.** Neko stores ⟨2νt S_ij⟩ and ⟨αt ∂θ/∂x_i⟩. They are converted on
  reading into SGS fluxes, `uw_sgs = −⟨2νt S13⟩`, `wt_sgs = −⟨αt ∂θ/∂z⟩`, so that
  `uw_tot = uw + uw_sgs` (only the deviatoric SGS stress is available).
* **Spectral accuracy.** Profiles and fields live on the GLL points. Vertical
  derivatives, integrals, and planes at a given height use the element
  polynomials (`sem.py`), not finite differences. The polynomial order is found
  automatically (duplicated element-interface heights in the CSV, header of the
  fld files) and stored in `ds.z.attrs["N"]`.
* **Field names.** fld slots are `u v w p t s1 s2 …`; name the passive scalars
  with `rename=`. In the example case: `s1` = Deardorff SGS TKE `e`, `s2` = LES
  filter width, `s3` = νt.
* **Time series.** Every diagnostic accepts a profile with dims `(z)` *or*
  `(time, z)`; with time it returns one value per statistics window.

## Plot style

Everything visual is in `nekabl/plotstyle.py`:

```python
nk.set_style(font_size=11, usetex=True, dpi=150, save_format="pdf", figure_dir="figs")
fig, axs = nk.subplots(1, 3)                     # package panel sizes
nk.plot_profiles(prof, "M", axs=axs[0])          # draw into existing axes ...
nk.plot_fluxes(prof, "wt", axs=axs[1])
nk.plot_budget(nk.tke_budget(prof), ax=axs[2])
nk.save(fig, "combined")
```

Visual encoding, the same in every plot:
* **colour = simulation** (or height / budget term when only one simulation is shown),
  fixed categorical order (colour-blind checked);
* **line style = component**: total `—`, resolved `– –`, SGS `···`, reference (MOST, k⁻⁵ᐟ³, Gaussian) dashed;
  with several simulations, heights in spectra/PDFs/correlations get line styles;
* `viridis` for positive fields, `RdBu_r` centred at 0 for signed ones;
* several simulations in slices / time-height plots -> one panel each, shared colour scale. Axis labels come
from `plotstyle.LABELS` (add your own variables there).

## Extending

Each module follows the same pattern, e.g. a new diagnostic in `diagnostics.py`:

```python
@_over_time                       # makes it work on (z) and (time, z) profiles
def my_height(prof, frac=0.1):
    z, f = _nodes(prof.tke)        # values on the grid points
    return _crossing(z, f, frac * f.max())
```

and a plot function follows this template (several simulations for free):

```python
def plot_my_profile(prof, var, zi=None, labels=None, ax=None):
    data = ps.as_dict(prof, labels)              # Dataset / list / dict -> {label: Dataset}
    fig, ax = ps.get_ax(ax, "profile")
    for i, (lab, ds) in enumerate(data.items()):
        ax.plot(ds[var], ps.zcoord(ds, zi), color=ps.color(i), ls=ps.ls("total"), label=lab)
    ax.set_xlabel(ps.label(var)); ps.zlabel(ax, zi)
    return fig, ax
```

## Science notes and references

* zi: minimum of the total heat flux (CBL), maximum of ∂Θ/∂z (Sullivan et al.
  1998, JAS), 5 % of the surface stress / 0.95 (Kosović & Curry 2000, JAS), TKE
  threshold, bulk Richardson number 0.25 (Vogelezang & Holtslag 1996, BLM).
* MOST functions as in Neko's wall model: Businger–Dyer (unstable),
  Beljaars & Holtslag 1991 (stable). `plot_similarity` shows φm, φh against MOST:
  the overshoot of φm above the first points is the log-layer mismatch
  (Brasseur & Wei 2010, PoF). `wall_model_check` reports the sampling height
  relative to z0, zi and L (Kawai & Larsson 2012, PoF) and compares u* from the
  LES stress with MOST applied to the mean wind.
* Resolved TKE ≳ 80 % of the total in the BL (Pope 2000) — needs the SGS TKE
  (Deardorff model: field files or scalar stats of the TKE scalar).
* TKE budget residual = SGS dissipation + SGS transport (+ transports if only
  the basic fluid set is available) + storage.
* Convective scaling: z/zi, variances/w*², TKE budget × zi/w*³; premultiplied
  spectra versus λ/zi peak at ~1–1.5 zi for w in the mixed layer.

## Limitations

* Field files: structured Cartesian box meshes (any element stretching),
  single-file fld output (no `nfiles > 1`), 3D only.
* Nek5000: `nek5000.KTH_FIELDS` is the variable order of the `stat.f` of the
  stable_16_TKE case (84 variables); another `stat.f` needs `fields_map=`. Physical
  parameters come from the .par/.usr (see METHODS.md) and can be overridden
  (`Run(case=dict(...))` or `--set`). The second-scalar statistics of that case do not contain
  the SGS TKE (see METHODS.md §7); the SGS TKE is taken from the field files.
* `avg_direction` other than `xy` is not handled (it is not useful for an ABL).

Checks: the fld reader returns exactly pymech's values (tests/), the GLL
derivative/integral/interpolation are exact for polynomials, the spectra
integrate to the variance, and the total-TKE budget of the example case closes
with the Deardorff dissipation (small residual). See METHODS.md.
