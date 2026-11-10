"""
logfile.py -- Parse the text log of a Neko (or Nek5000) run.

    steps, meta = read_log("output/log.out")

`steps` is a pandas DataFrame with one row per time step:
    step, t, dt, cfl, step_time (wall-clock s), elapsed (s),
    it_<solver> (Krylov iterations, e.g. it_Pressure), res_<solver>,
    wm_Ri_b wm_L_ob wm_utau wm_magu wm_ti wm_ts wm_q wm_bc_value: the "Wall model
    diagnostics" section of the MOST / Richardson wall models (SUMS over the wall
    nodes, except bc_value; see wall_model_means)
`meta` is a dict with version, MPI ranks, backend, precision, elements,
warnings, whether the run ended normally, ...

Only regular expressions on the standard log lines are used, so a log that is
still being written (running job) can be read too.
"""

import re
from collections import Counter

import numpy as np

_FLOAT = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"

# Neko
_STEP = re.compile(r"Step\s*=\s*(\d+)\s+t\s*=\s*(" + _FLOAT + ")")
_CFL = re.compile(r"CFL:\s*(" + _FLOAT + r")\s+dt:\s*(" + _FLOAT + ")")
_KSP = re.compile(r"^\s*(\d+)\s*\|\s*(.+?)\s+(\d+)\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*$")
_STEPTIME = re.compile(r"Total time for step\s+\d+\s*\(s\):\s*(" + _FLOAT + ")")
_ELAPSED = re.compile(r"(?:Total elapsed time|Elapsed time) \(s\):\s*(" + _FLOAT + ")")
# Nek5000
_NEK_STEP = re.compile(r"Step\s+(\d+),\s*t=\s*(" + _FLOAT + r"),\s*DT=\s*(" + _FLOAT + r"),\s*C=\s*("
                       + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + ")")
_WM = re.compile(r"^\s*(Ri_b|L_ob|utau|magu|ti|ts|q|bc_value):\s*(" + _FLOAT + ")")
_NEK_PRES = re.compile(r"^\s*(\d+)\s+U-PRES gmres\s*:?\s*(\d+)")

_META = {
    "version": re.compile(r"\(version:\s*(.+?)\)"),
    "start": re.compile(r"Start time:\s*(.+)$"),
    "running_on": re.compile(r"Running on:\s*(.+)$"),
    "backend": re.compile(r"Bcknd type:\s*(.+)$"),
    "precision": re.compile(r"Real type\s*:\s*(.+)$"),
    "elements": re.compile(r"(?:Elements\s*:|nelements\s*=)\s*(\d+)"),
}


def _f(s):
    return float(s.replace("D", "E").replace("d", "e"))


def read_log(path):
    """Parse a Neko/Nek5000 log file. Returns (steps DataFrame, meta dict)."""
    import pandas as pd

    rows, cur = [], None
    meta = {k: None for k in _META}
    warns = Counter()
    meta["errors"] = []
    meta["normal_end"] = False
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = _STEP.search(line)
            if m and "Step =" in line:
                cur = dict(step=int(m.group(1)), t=_f(m.group(2)))
                rows.append(cur)
                continue
            m = _NEK_STEP.search(line)
            if m:
                cur = dict(step=int(m.group(1)), t=_f(m.group(2)), dt=_f(m.group(3)),
                           cfl=_f(m.group(4)), elapsed=_f(m.group(5)), step_time=_f(m.group(6)))
                rows.append(cur)
                continue
            if cur is not None:
                m = _CFL.search(line)
                if m:
                    cur["cfl"], cur["dt"] = _f(m.group(1)), _f(m.group(2))
                    continue
                m = _KSP.match(line)
                if m:
                    name = m.group(2).strip().replace(" ", "_")
                    cur["it_" + name] = int(m.group(3))
                    cur["res_" + name] = _f(m.group(5))
                    continue
                m = _NEK_PRES.match(line)
                if m:
                    cur["it_Pressure"] = int(m.group(2))
                    continue
                m = _WM.match(line)
                if m:     # Neko wall model diagnostics: sums over the wall nodes
                    cur["wm_" + m.group(1)] = _f(m.group(2))
                    continue
                m = _STEPTIME.search(line)
                if m:
                    cur["step_time"] = _f(m.group(1))
                    continue
                m = _ELAPSED.search(line)
                if m:
                    cur["elapsed"] = _f(m.group(1))
                    continue
            for k, rx in _META.items():
                if meta[k] is None:
                    m = rx.search(line)
                    if m:
                        meta[k] = m.group(1).strip()
            if "WARNING" in line:
                warns[line.strip().strip("*").strip()] += 1
            if "ERROR" in line:
                meta["errors"].append(line.strip())
            if "Normal end" in line or "run successful" in line:
                meta["normal_end"] = True
    steps = pd.DataFrame(rows)
    if not steps.empty:
        steps = steps.drop_duplicates("step", keep="last").set_index("step")
        if "dt" not in steps and len(steps) > 1:
            steps["dt"] = np.diff(steps["t"].values, prepend=np.nan)
    meta["warnings"] = dict(warns)
    meta["file"] = path
    return steps, meta


def describe(steps, meta):
    """Short text summary of a parsed log."""
    L = []
    add = L.append
    add(f"log file       : {meta['file']}")
    for k in ("version", "start", "running_on", "backend", "precision", "elements"):
        if meta.get(k):
            add(f"{k:<15}: {meta[k]}")
    if steps.empty:
        add("no time steps found")
        return "\n".join(L)
    add(f"time steps     : {steps.index[0]} ... {steps.index[-1]}  ({len(steps)} logged)")
    add(f"simulated time : t = {steps.t.iloc[0]:.4g} ... {steps.t.iloc[-1]:.4g}")
    if "dt" in steps:
        dt = steps.dt.dropna()
        add(f"dt             : mean {dt.mean():.4g}, min {dt.min():.4g}, max {dt.max():.4g}")
    if "cfl" in steps:
        add(f"CFL            : mean {steps.cfl.mean():.3f}, max {steps.cfl.max():.3f}")
    if "step_time" in steps:
        st = steps.step_time.dropna()
        add(f"wall time/step : mean {st.mean():.3g} s (median {st.median():.3g} s)")
    if "elapsed" in steps and steps.elapsed.notna().any():
        el = steps.elapsed.dropna().iloc[-1]
        add(f"elapsed        : {el:.1f} s = {el / 3600:.2f} h")
        if "dt" in steps:
            simt = steps.t.iloc[-1] - steps.t.iloc[0]
            if simt > 0:
                add(f"speed          : {simt / el:.3g} simulated s per wall s")
    for c in [c for c in steps.columns if c.startswith("it_")]:
        add(f"{c:<15}: mean {steps[c].mean():.1f}, max {steps[c].max():.0f} iterations")
    if meta["warnings"]:
        add(f"warnings       : {sum(meta['warnings'].values())} "
            f"({len(meta['warnings'])} distinct)")
        for w, n in list(meta["warnings"].items())[:5]:
            add(f"   {n:>6} x  {w[:90]}")
    if meta["errors"]:
        add(f"errors         : {meta['errors'][:3]}")
    add(f"normal end     : {meta['normal_end']}")
    return "\n".join(L)


def wall_model_means(steps, n_wall):
    """Wall-model diagnostics of the log as horizontal means: Neko prints the sum
    over the n_wall wall nodes (GLL points of the wall faces, counted once per
    element: n_wall = (wall elements) x (N+1)^2, see Run.n_wall_nodes).
    Returns a DataFrame indexed by step with t, ustar, q, ts (wall-node temperature),
    ti (temperature at h), magu, L, Ri_b, bc_value."""
    import pandas as pd

    cols = {"wm_utau": "ustar", "wm_q": "q", "wm_ts": "ts", "wm_ti": "ti", "wm_magu": "magu",
            "wm_L_ob": "L", "wm_Ri_b": "Ri_b"}
    if "wm_utau" not in steps:
        return None
    out = pd.DataFrame({"t": steps["t"]})
    for k, v in cols.items():
        if k in steps:
            out[v] = steps[k] / n_wall
    if "wm_bc_value" in steps:
        out["bc_value"] = steps["wm_bc_value"]
    return out.dropna(subset=["ustar"])
