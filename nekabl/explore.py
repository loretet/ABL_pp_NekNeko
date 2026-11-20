"""
explore.py -- Comprehensive data exploration from the command line.

Runs (almost) everything in the package on one or more simulations and saves
the figures + a text summary, e.g. at the end of a job script:

    nekabl-explore /path/to/case                        # one Neko run
    nekabl-explore runA runB --labels coarse fine       # compare runs
    nekabl-explore runA --t0 2000 --t1 3000 --out figs --format pdf
    nekabl-explore runA --t0 5000 --evolution            # + thin profiles of each window
    nekabl-explore runA --t0 8000 --evolution 5000 10000  # mean 8000-end, thin 5000-10000
    nekabl-explore nek_case --set T0=263.5,z0=0.1,f=1.39e-4   # Nek5000 parameters
    nekabl-explore --config runs.json                   # per-run options (JSON)

runs.json:
    {"runs": [{"path": "neko_run", "label": "Neko", "rename": {"s3": "nut"}},
              {"path": "nek_run", "label": "Nek5000",
               "case": {"T0": 263.5, "z0": 0.1, "f": 1.39e-4}}],
     "t0": 2000, "out": "figs"}

Each figure is produced independently: if something is not available
(e.g. 'basic' statistics, no field files, no log) it is skipped and the
reason is printed and written to the summary. From Python: nk.explore(...).
"""

import argparse
import json
import os
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import diagnostics as dg  # noqa: E402
from . import plotstyle as ps  # noqa: E402
from . import profiles as pr  # noqa: E402
from . import turbulence as tb  # noqa: E402
from . import visualization as vz  # noqa: E402
from .run import Run  # noqa: E402


class _Log:
    def __init__(self, path):
        self.path = path
        self.lines = []

    def __call__(self, msg=""):
        print(msg)
        self.lines.append(str(msg))

    def write(self):
        with open(self.path, "w") as fh:
            fh.write("\n".join(self.lines) + "\n")


def explore(runs, labels=None, t0=None, t1=None, out="nekabl_figures", fmt="png", zmax=None,
            use_fields=True, last_fields=4, rename=None, case=None, verbose_errors=False,
            evolution=None, scalars=None):
    """Make all figures for one or more runs (paths or Run objects).

    evolution : None (off), True (thin profiles of every statistics window of
                the averaging window) or (t_start, t_end). One run only.
    scalars   : None (all scalars) or names / symbols (['co2', 'temperature']) for
                the scalar and entrainment figures."""
    ps.set_style(figure_dir=out, save_format=fmt)
    os.makedirs(out, exist_ok=True)
    log = _Log(os.path.join(out, "summary.txt"))
    labels = labels or [None] * len(runs)
    R = []
    for r, lab in zip(runs, labels):
        R.append(r if isinstance(r, Run) else Run(r, label=lab, rename=rename, case=case))
    saved = []

    def do(name, func):
        try:
            fig = func()
            if fig is None:
                return
            path = ps.save(fig, name)
            plt.close(fig)
            saved.append(path)
            print(f"  [ok] {os.path.basename(path)}")
        except Exception as err:  # noqa: BLE001 - keep going whatever happens
            plt.close("all")
            log(f"  [skipped] {name}: {type(err).__name__}: {err}")
            if verbose_errors:
                traceback.print_exc()

    # ------------------------------------------------------------------ summary
    from .summary import summarize
    for r in R:
        if r.kind == "neko":
            log(summarize(r.path, output=r.out, print_it=False))
        else:
            from .case import describe
            from .nek5000 import find_sts
            log(f"==== {r.label}: Nek5000 case {r.path}")
            log(describe(r.case))
            sts = find_sts(r.out)
            log(f"statistics files: {len(sts)}; field files '{r.field_prefix}': {len(r.field_files())}")
    profs, hists, tss = {}, {}, {}
    for r in R:
        try:
            profs[r.label] = r.profile(t0, t1)
            hists[r.label] = r.history()
            tss[r.label] = r.timeseries()
            p = profs[r.label]
            log(f"---- {r.label}: averaging {p.attrs['t0']:.0f}-{p.attrs['t1']:.0f} s, "
                f"fluid set '{p.attrs.get('fluid_set', '?')}'")
            log("     " + ", ".join(f"{k} = {p.attrs[k]:.4g}" for k in
                                    ("ustar", "wt_s", "L", "zi", "wstar", "zi_L") if k in p.attrs))
        except Exception as err:  # noqa: BLE001
            log(f"---- {r.label}: no statistics ({err})")
    if not profs:
        log.write()
        return saved
    evo = None
    if evolution is not None and evolution is not False:
        if len(profs) > 1:
            log("     profile evolution: only for a single simulation, ignored")
        else:
            r = next(r for r in R if r.label in profs)
            p = profs[r.label]
            ta, tb_ = (p.attrs["t0"], p.attrs["t1"]) if evolution is True else evolution
            try:
                evo = r.evolution(ta, tb_)
                log(f"     profile evolution: {evo.time.size} statistics windows, "
                    f"t = {ta:.0f}-{float(evo.time[-1]):.0f} s")
            except Exception as err:  # noqa: BLE001
                log(f"     profile evolution: {err}")
    user_zmax = zmax
    zi_max = max(p.attrs.get("zi", np.nan) for p in profs.values())
    ztop = min(float(p.z.max()) for p in profs.values())
    zmax = zmax or (min(1.6 * zi_max, ztop) if np.isfinite(zi_max) else ztop)
    # z/zi plots: each run is scaled with its own zi -> top = 1.6 zi or the lowest domain top
    zis_ = [(float(p.z.max()), p.attrs.get("zi", np.nan)) for p in profs.values()]
    if user_zmax is not None and np.isfinite(zi_max):
        zmx = user_zmax / zi_max
    elif all(np.isfinite(z) and z > 0 for _, z in zis_):
        zmx = min([1.6] + [top / z for top, z in zis_])
    else:
        zmx = None
    one = len(profs) == 1

    # --------------------------------------------------------------- statistics
    print("statistics figures:")
    for var in ("wt_tot", "tke", "M"):
        do(f"01_time_height_{var}", lambda var=var: pr.plot_time_height(
            {k: h for k, h in hists.items() if var in h}, var,
            zi={k: t.zi for k, t in tss.items()}, zmax=zmax)[0])
    do("02_timeseries", lambda: dg.plot_timeseries(
        tss, [v for v in ("ustar", "wt_s", "L", "zi", "zi_L") if v in next(iter(tss.values()))])[0])
    do("03_momentum", lambda: pr.plot_overview(profs, "momentum", zi=True, zmax=zmx, evolution=evo)[0])
    do("03_hodograph", lambda: pr.plot_hodograph(profs, zmax=zmax, evolution=evo)[0])
    do("04_heat", lambda: pr.plot_overview(profs, "heat", zi=True, zmax=zmx, evolution=evo)[0])
    from . import entrainment as en

    def _sel(p):
        try:
            return en.resolve_scalars(p, scalars)
        except KeyError:
            return []
    for s in sorted({c for p in profs.values() for c in _sel(p) if c != "t"}):
        do(f"05_scalar_{s}", lambda s=s: pr.plot_overview(
            {k: p for k, p in profs.items() if s in p.attrs.get("scalars", "")}, "scalar",
            sym=s, zi=True, zmax=zmx, evolution=evo)[0])
    do("06_stability", lambda: pr.plot_overview(profs, "stability", zi=True, zmax=zmx, evolution=evo)[0])
    do("02_ekman_transport", lambda: dg.plot_ekman(tss)[0])
    for k, p in profs.items():
        if "Q_cross" in p.attrs:
            log(f"     {k}: cross-isobaric transport Q = {p.attrs['Q_cross']:.4g} m2/s"
                + (f" (Ekman, -tau_s/f: {p.attrs['Q_cross_ekman']:.4g})" if "Q_cross_ekman" in p.attrs else "")
                + f", along-isobaric deficit {p.attrs['Q_along']:.4g} m2/s, turning angle surface "
                f"{p.attrs['alpha_s']:+.1f} deg" + (f", zi {p.attrs['alpha_zi']:+.1f} deg" if "alpha_zi" in p.attrs else "")
                + f", wind direction surface {p.attrs['wdir_s']:.0f} deg, geostrophic {p.attrs['wdir_g']:.0f} deg")
    do("07_sgs", lambda: dg.plot_sgs(profs, zi=True, zmax=zmx, evolution=evo)[0])

    def _sim():
        sims = {k: dg.similarity(p, p.attrs, r.case) for (k, p), r in zip(profs.items(), R)}
        h = None
        if one and R[0].case.h_index is not None:
            h = float(next(iter(profs.values())).z[R[0].case.h_index])
        sev = None
        if evo is not None:
            try:
                sev = dg.similarity_evolution(evo, R[0].case)
            except Exception as err:  # noqa: BLE001
                log(f"     similarity evolution: {err}")
        return dg.plot_similarity(sims, zmax_zi=0.4, h=h, evolution=sev)[0]

    do("08_most_similarity", _sim)
    for r in R:
        if r.case.wall_model and r.label in profs:
            try:
                import contextlib
                import io
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    dg.wall_model_check(profs[r.label], r.case, profs[r.label].attrs)
                log(f"{r.label}:\n" + buf.getvalue())
            except Exception as err:  # noqa: BLE001
                log(f"  [skipped] wall model check {r.label}: {err}")
    def _surf():
        chk = {r.label: dg.surface_check(hists[r.label], r.case) for r in R if r.label in hists}
        ser = {}
        for r in R:
            try:
                s = r.wall_model_series()
                if s is not None:
                    ser[r.label] = s
            except Exception as err:  # noqa: BLE001
                log(f"     {r.label}: no per-step wall-model output ({err})")
        return dg.plot_surface(chk, ser)[0]

    do("08_surface_heat_flux_temperature", _surf)
    do("09_eddy_viscosity", lambda: tb.plot_eddy_viscosity(
        {k: tb.eddy_viscosity(p) for k, p in profs.items()}, zi=True, zmax=zmx, evolution=evo)[0])

    # -------------------------------------------------------------- entrainment
    print("entrainment figures:")
    ents, ebud = {}, {}
    for r in R:
        if r.label not in profs:
            continue
        p = profs[r.label]
        if not p.attrs.get("wt_s", 1.0) > 0:
            log(f"  [skipped] entrainment {r.label}: surface heat flux <= 0 (stable BL: no mixed layer "
                "capped by an inversion; for a statically stable layer use the plain profile figures)")
            continue
        try:
            e = r.entrainment(p.attrs["t0"], p.attrs["t1"], scalars=scalars)
            ents[r.label] = e
            log(f"---- {r.label}: " + en.entrainment_table(e))
            if e.time.size >= 4:
                ebud[r.label] = en.entrainment_budget(r.evolution(p.attrs["t0"], p.attrs["t1"]), e,
                                                      scalars)
        except Exception as err:  # noqa: BLE001
            log(f"  [skipped] entrainment {r.label}: {type(err).__name__}: {err}")
    if ents:
        avg = {k: p for k, p in profs.items() if k in ents}
        has = lambda p, c: c in _sel(p)  # noqa: E731
        do("18_inversion_momentum_heat", lambda: en.plot_inversion(
            avg, scalars=["t"], momentum=True, ent={k: e for k, e in ents.items()})[0])
        sc_ = [c for c in sorted({c for p in avg.values() for c in _sel(p)}) if c != "t"]
        if sc_:
            for c in sc_:
                sub = {k: p for k, p in avg.items() if has(p, c)}
                do(f"18_inversion_scalar_{c}", lambda c=c, sub=sub: en.plot_inversion(
                    sub, scalars=[c], ent={k: ents[k] for k in sub})[0])
        do("18_entrainment_timeseries", lambda: en.plot_entrainment(ents, scalars=scalars)[0])
    if ebud:
        do("18_entrainment_budget", lambda: en.plot_entrainment_budget(ebud, scalars=scalars)[0])

    # ------------------------------------------------------------------ fields
    flds = {}
    if use_fields:
        for r in R:
            try:     # snapshots inside the averaging window (else the last one)
                tmin = profs[r.label].attrs.get("t0") if r.label in profs else None
                try:
                    flds[r.label] = r.fields(tmin=tmin, last=last_fields)
                except FileNotFoundError:
                    flds[r.label] = r.fields(last=1)
                    log(f"     {r.label}: no field file inside the averaging window, using the last one")
                log(f"---- {r.label}: {flds[r.label].time.size} field snapshots, "
                    f"t = {flds[r.label].time.values[0]:.0f}-{flds[r.label].time.values[-1]:.0f} s")
            except Exception as err:  # noqa: BLE001
                log(f"---- {r.label}: no field files ({err})")

    # budgets (dissipation from the fields when possible)
    eps = {}
    for k, f in flds.items():
        try:
            eps[k] = tb.dissipation_from_fields(f, next(r.case for r in R if r.label == k))
            log(f"     {k}: SGS dissipation from fields ({eps[k].attrs['model']} model)")
        except Exception as err:  # noqa: BLE001
            log(f"     {k}: no dissipation from fields ({type(err).__name__}: {err})")
    tke = {k: tb.tke_budget(p, eps=eps.get(k)) for k, p in profs.items()}
    for k, b in tke.items():
        if b.attrs.get("missing"):
            log(f"     TKE budget {k}: missing {b.attrs['missing']}")
    scale = {k: p.attrs["zi"] / p.attrs["wstar"] ** 3 if p.attrs.get("wstar", 0) > 0
             else p.attrs["zi"] / p.attrs["ustar"] ** 3 for k, p in profs.items()}
    do("10_tke_budget", lambda: tb.plot_budget(tke, zi=True, scale=scale, zmax=zmx)[0])
    do("10_heat_flux_budget", lambda: tb.plot_budget(
        {k: tb.heat_flux_budget(p) for k, p in profs.items()}, zi=True, zmax=zmx)[0])
    do("10_theta_variance_budget", lambda: tb.plot_budget(
        {k: tb.theta_variance_budget(p) for k, p in profs.items()}, zi=True, zmax=zmx)[0])

    if flds:
        print("field figures:")
        zis = {k: profs[k].attrs["zi"] for k in flds if k in profs}
        for k, f in flds.items():
            zi = zis.get(k, float(f.z.max()) / 2)
            safe = k.replace(" ", "_").replace("/", "_")
            do(f"11_hslice_w_{safe}", lambda f=f, zi=zi: vz.plot_hslices(
                f, "w", [0.1 * zi, 0.5 * zi, 0.9 * zi])[0])
            do(f"11_hslices_theta_{safe}", lambda f=f, zi=zi: vz.plot_hslices(
                f, "t", [0.1 * zi, 0.5 * zi, 0.9 * zi], fluct=True)[0])
            do(f"12_vslice_w_{safe}", lambda f=f: vz.plot_vslice(
                f, "w", y=float(f.y.mean()), zmax=zmax)[0])

            def _ff(f=f, k=k):
                extra = [v for v in ("e", "nut") if v in f]
                pf = pr.derive(pr.time_average(pr.from_fields(f, extra=extra)),
                               case=next(r.case for r in R if r.label == k))
                pf.attrs["label"] = "snapshots"
                r = next(r for r in R if r.label == k)      # statistics over the snapshot period
                start = r.raw.time.values - r.raw.window.values
                cand = start[start <= float(f.time[0]) + 1e-6]
                ps_ = r.profile(float(cand.max()) if cand.size else None, None)
                return pr.plot_profiles({"statistics": ps_, "snapshots": pf},
                                        [v for v in ("M", "T", "tke", "wt", "tke_sgs") if v in pf],
                                        zmax=zmax)[0]
            do(f"13_stats_vs_snapshots_{safe}", _ff)
        hz = {k: [0.1 * z, 0.5 * z, 0.9 * z] for k, z in zis.items()}

        def _spec(var, direction, pm):
            sp = {k: tb.spectrum(f, var, hz[k], direction) for k, f in flds.items() if k in hz}
            return tb.plot_spectra(sp, premultiplied=pm, zi=True if pm else None)[0]

        do("14_spectra_w_premultiplied", lambda: _spec("w", "radial", True))
        do("14_spectra_u_x", lambda: _spec("u", "x", False))
        do("14_spectra_t_radial", lambda: _spec("t", "radial", False))
        do("15_autocorrelation_w", lambda: tb.plot_autocorrelation(
            {k: tb.autocorrelation(f, "w", hz[k], "x") for k, f in flds.items() if k in hz}, zi=True)[0])
        do("16_pdf_w", lambda: tb.plot_pdf({k: tb.pdf(f, "w", hz[k]) for k, f in flds.items() if k in hz})[0])
        for k, f in flds.items():
            if k in zis:
                safe = k.replace(" ", "_").replace("/", "_")
                do(f"16_joint_pdf_wt_{safe}", lambda f=f, z=zis[k]: tb.plot_joint_pdf(f, "w", "t", 0.3 * z)[0])
                q = tb.quadrant_analysis(f, "w", "t", [0.3 * zis[k]]).flux_fraction.isel(z=0).values
                log(f"     {k}: w-theta quadrant flux fractions at 0.3 zi: "
                    + ", ".join(f"Q{i + 1} {v:+.2f}" for i, v in enumerate(q)))
                L = tb.integral_length(tb.autocorrelation(f, "w", hz[k], "x")).values
                log(f"     {k}: integral length of w along x at 0.1/0.5/0.9 zi: "
                    + ", ".join(f"{v:.0f} m" for v in L))
                log(f"     {k}: largest dt for CFL 0.5 on the last snapshot: {dg.cfl_dt(f, 0.5):.3g} s")

        for k, f in flds.items():
            if k not in zis or k not in ents:     # no inversion (stable BL) -> nothing to do
                continue
            safe = k.replace(" ", "_").replace("/", "_")
            try:
                names = [n for n in next(r for r in R if r.label == k).scalar_fields(scalars) if n in f]
                h = en.inversion_height_map(f, zis[k])
                st = en.inversion_stats(h, zis[k])
                log(f"     {k}: local inversion height h(x,y): mean {st['h_mean']:.0f} m, std "
                    f"{st['h_std']:.0f} m ({100 * st['h_std_over_h']:.1f} % of zi), skewness "
                    f"{st['h_skew']:+.2f}" + (f"; Eulerian EZ depth {float(ents[k].ez.mean()):.0f} m"
                                              if k in ents else ""))
                do(f"19_inversion_map_{safe}", lambda h=h, zi=zis[k]: en.plot_inversion_map(h, zi=zi)[0])
                comp = en.interface_composite(f, h, names)
                do(f"19_interface_composite_{safe}", lambda comp=comp: en.plot_interface_composite(comp)[0])
                for c, qd in en.quadrants_at_inversion(f, zis[k], names).items():
                    q = qd.flux_fraction.isel(z=len(qd.z) // 2).values
                    log(f"     {k}: w'-{c}' quadrant flux fractions at zi: "
                        + ", ".join(f"Q{i + 1} {v:+.2f}" for i, v in enumerate(q)))
            except Exception as err:  # noqa: BLE001
                log(f"  [skipped] interface analysis {k}: {type(err).__name__}: {err}")

    # --------------------------------------------------------------------- log
    logs = {}
    for r in R:
        try:
            lg = r.log()
            if lg is not None and not lg[0].empty:
                logs[r.label] = lg[0]
        except Exception as err:  # noqa: BLE001
            log(f"  [skipped] log of {r.label}: {err}")
    if logs:
        do("17_timestep", lambda: dg.plot_timestep(logs)[0])
    steps = {}
    for r in R:
        try:
            s = r.step_series(every=10)
            if s is not None:
                steps[r.label] = s
        except Exception as err:  # noqa: BLE001
            log(f"  [skipped] step series of {r.label}: {err}")
    if steps:
        do("17_step_series", lambda: dg.plot_step_series(steps)[0])
    log(f"\n{len(saved)} figures written to {os.path.abspath(out)}")
    log.write()
    return saved
