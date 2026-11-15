"""
run.py -- One object per simulation, the same for Neko and Nek5000.

    a = nk.Run("cases/neutral", label="neutral")          # Neko (has a .case)
    b = nk.Run("cases/gabls_nek", label="Nek5000")        # Nek5000 (has a .par / SIZE)

    prof = {r.label: r.profile(2000, 3000) for r in (a, b)}
    nk.plot_profiles(prof, ["M", "T"], zi=True)            # z/zi of each run

Everything a Run returns is in the package convention (z vertical, Neko
names), so every function of the package works on both codes:

    run.case          physical parameters (CaseInfo; see case.py)
    run.raw           raw statistics (time, z)                    [cached]
    run.profile(t0, t1)   time-averaged derived profiles (z), with the bulk
                      scales (u*, L, zi, w*, ...) in .attrs        [cached]
    run.history()     derived profiles for every statistics window (time, z)
    run.timeseries()  u*, L, zi, w* ... per statistics window
    run.fields(...)   field snapshots (time, z, y, x)
    run.log()         (steps, meta) from the log file, or None

Default averaging window of profile(): the second half of the statistics.
"""

import glob
import os

import numpy as np

from . import case as _case
from . import fields as _fields
from . import plotstyle as _ps
from . import stats as _stats


def detect_kind(path):
    """'neko' if a .case file is found, 'nek5000' if a .par/.rea/.usr/SIZE file is.
    A .case in the folder itself wins, then Nek5000 markers, then a .case in
    the parent (Neko output sub-folders)."""
    if glob.glob(os.path.join(path, "*.case")):
        return "neko"
    for pat in ("*.par", "*.rea", "*.re2", "SIZE", "*.usr", os.path.join("compile", "SIZE"),
                os.path.join("compile", "*.usr")):
        if glob.glob(os.path.join(path, pat)):
            return "nek5000"
    if _case.find_case(path):
        return "neko"
    raise FileNotFoundError(f"{path}: neither a Neko (.case) nor a Nek5000 (.par/.rea/SIZE) case")


class Run:
    """A simulation folder (Neko or Nek5000). See module docstring.

    Parameters
    ----------
    path         : case folder.
    label        : name in legends (default: folder name).
    kind         : 'neko' / 'nek5000' (default: detected).
    output       : output sub-folder (Neko default 'output' if it exists).
    scalars      : Neko {field name: symbol}, default {'temperature': 't'}.
    rename       : field-file slot names, e.g. {'s1': 'e', 's3': 'nut'}.
    field_prefix : field-file prefix (Neko default 'field0').
    case         : CaseInfo or dict to override / complete the parameters
                   (needed for Nek5000: g, T0, z0, f ... are in the .usr).
    start_time   : statistics start time (first window weight).
    nek          : Nek5000 options, see stats.load_nek5000_stats.
    """

    def __init__(self, path, label=None, kind=None, output=None, scalars=None, rename=None,
                 field_prefix=None, case=None, start_time=None, nek=None):
        self.path = os.path.abspath(path)
        self.label = label or os.path.basename(self.path.rstrip("/"))
        self.kind = kind or detect_kind(self.path)
        self.scalars = dict(scalars) if scalars else {"temperature": "t"}
        self._auto_scalars = scalars is None
        self.rename = rename or {}
        self.nek = nek or {}
        self.start_time = start_time
        if self.kind == "neko":
            cpath = _case.find_case(self.path)
            self.case = _case.read_case(cpath) if cpath else _case.case_from_dict({})
            out = os.path.join(self.path, output or "output")
            self.out = out if os.path.isdir(out) else self.path
            self.field_prefix = field_prefix or "field0"
            self.vertical = "z"
            if rename is None:     # Neko writes the 1st scalar in 't', the others in s1, s2 ...
                for i, name in enumerate((self.case.scalars or [])[1:]):
                    self.rename[f"s{i + 1}"] = "e" if name.lower() == "tke" else name
            if self._auto_scalars:   # passive scalars with statistics files: one free letter each
                self._detect_scalars()
        else:
            from . import nek5000
            self.case = nek5000.read_par(self.path)
            out = os.path.join(self.path, output or "output")
            self.out = out if os.path.isdir(out) else self.path
            self.field_prefix = field_prefix or nek5000.guess_field_prefix(
                self.out, nek5000.casename(self.path))
            self.vertical = self.nek.get("vertical", "y")
            if rename is None:     # Nek5000 + SGS TKE model: scalar 1 is the SGS TKE
                self.rename["s1"] = "e"
        if case is not None:
            upd = case if isinstance(case, dict) else vars(case)
            for k, v in upd.items():
                if v is not None:
                    setattr(self.case, k, v)
        self._raw = None
        self._cache = {}

    def _detect_scalars(self):
        """Add the passive scalars of the .case that have statistics files."""
        pool = [c for c in "qcrmbfghjlo" if c not in self.scalars.values()]
        for name in (self.case.scalars or [])[1:]:
            if name.lower() in ("tke", "e", "nut") or name in self.scalars or not pool:
                continue
            if _stats._first_existing(self.out, [f"scalar_stats_{name}"]):
                self.scalars[name] = pool.pop(0)

    @property
    def scalar_names(self):
        """{symbol: name} of the scalars of the statistics ('t' = temperature)."""
        return {sym: name for name, sym in self.scalars.items()}

    def scalar_fields(self, scalars=None):
        """Names of the scalars in the field files ('t' for the temperature)."""
        names = ["t"] + [n for n in self.scalars if self.scalars[n] != "t"]
        if scalars is None:
            return names
        from .entrainment import resolve_scalars
        return [("t" if s == "t" else self.scalar_names[s]) for s in
                resolve_scalars({"scalars": "".join(self.scalars.values()),
                                 "scalar_names": self.scalar_names}, scalars)]

    def __repr__(self):
        return f"Run('{self.label}', {self.kind}, {self.path})"

    # -- statistics -----------------------------------------------------------
    @property
    def raw(self):
        if self._raw is None:
            if self.kind == "neko":
                self._raw = _stats.load_stats(self.out, scalars=self.scalars, case=self.case,
                                              start_time=self.start_time, verbose=False)
            else:
                from . import nek5000
                self._raw = nek5000.load_stats(self.out, case=self.case, start_time=self.start_time,
                                               **self.nek)
            self._raw.attrs["label"] = self.label
        return self._raw

    def _window(self, t0, t1):
        t = self.raw.time.values
        if t0 is None:
            start = t[0] - self.raw.window.values[0]
            t0 = start + 0.5 * (t[-1] - start)
            t0 = (t - self.raw.window.values)[np.argmin(abs(t - self.raw.window.values - t0))]
        return t0, t1

    def profile(self, t0=None, t1=None, zi_method="auto", zmax=None, moments="window"):
        """Time-averaged derived profiles over [t0, t1], with the bulk scales
        (ustar, wt_s, thetastar, L, zi, wstar, zi_L, ...) in .attrs.
        moments: 'window' (fluctuations about the mean of each statistics
        window) or 'global' (about the mean of [t0, t1]); see time_average."""
        from .diagnostics import scales
        from .profiles import derive, time_average

        t0, t1 = self._window(t0, t1)
        key = ("profile", t0, t1, zi_method, zmax, moments)
        if key not in self._cache:
            prof = derive(time_average(self.raw, t0, t1, moments), case=self.case)
            try:
                from .diagnostics import ekman_transport
                sc = scales(prof, self.case, zi_method=zi_method, zmax=zmax)
                prof.attrs.update({k: float(v) for k, v in sc.items()})
                prof.attrs.update(ekman_transport(prof, self.case))
            except (KeyError, ValueError) as err:
                print(f"[nekabl] {self.label}: no bulk scales ({err})")
            prof.attrs["label"] = self.label
            prof.attrs["scalar_names"] = self.scalar_names
            for sym, name in self.scalar_names.items():
                _ps.add_scalar_labels(sym, name)
            self._cache[key] = prof
        return self._cache[key]

    def history(self):
        """Derived profiles for every statistics window (time, z)."""
        from .profiles import derive

        if "history" not in self._cache:
            h = derive(self.raw, case=self.case)
            h.attrs["label"] = self.label
            h.attrs["scalar_names"] = self.scalar_names
            self._cache["history"] = h
        return self._cache["history"]

    def evolution(self, t0=None, t1=None):
        """Derived profiles of every statistics window inside [t0, t1] (time, z),
        for the thin 'evolution' lines of the profile plots. Default: the
        averaging window of profile(). Each profile is the average over one
        statistics window (the output interval of the case)."""
        t0, t1 = self._window(t0, t1)
        h = self.history()
        start = h.time.values - h.window.values
        tol = 1e-6 * max(1.0, float(abs(h.time.values).max()))
        keep = start >= t0 - tol
        if t1 is not None:
            keep &= h.time.values <= t1 + tol
        if not keep.any():
            raise ValueError(f"{self.label}: no statistics window inside [{t0}, {t1}]")
        ev = h.isel(time=np.where(keep)[0])
        ev.attrs = dict(h.attrs)
        return ev

    def entrainment(self, t0=None, t1=None, scalars=None, momentum=True, budget=False, **kw):
        """Entrainment diagnostics of every statistics window in [t0, t1]
        (default: the averaging window): inversion height and depth, jumps,
        entrainment fluxes and velocities of momentum, heat and the passive
        scalars (`scalars`: None = all, or names, e.g. ['co2', 'temperature']).
        See nekabl.entrainment.entrainment_rates for the options (method,
        zmax, frac, ml, ft_depth). budget=True returns (ent, budget) with the
        growth-based estimates (needs >= 4 windows)."""
        from .entrainment import entrainment_budget, entrainment_rates

        prof = self.evolution(t0, t1)
        ent = entrainment_rates(prof, scalars, momentum, **kw)
        ent.attrs["label"] = self.label
        if not budget:
            return ent
        b = entrainment_budget(prof, ent, scalars)
        b.attrs["label"] = self.label
        return ent, b

    def timeseries(self, zi_method="auto", zmax=None):
        """u*, L, zi, w*, ... for every statistics window."""
        from .diagnostics import scales

        ts = scales(self.history(), self.case, zi_method=zi_method, zmax=zmax)
        try:      # cross-isobaric transport and wind turning (zi of every window)
            from .diagnostics import ekman_transport
            h = self.history()
            ek = [ekman_transport(h.isel(time=i).assign_attrs(zi=float(ts.zi[i])), self.case)
                  for i in range(h.time.size)]
            for k in ek[0]:
                ts[k] = ("time", [e[k] for e in ek])
        except (KeyError, ValueError, AttributeError) as err:
            print(f"[nekabl] {self.label}: no Ekman transport ({err})")
        ts.attrs["label"] = self.label
        return ts

    # -- fields and log ----------------------------------------------------------
    def field_files(self):
        """[(path, time)] of the field files."""
        return _fields.list_fields(self.out, self.field_prefix)

    def fields(self, tmin=None, tmax=None, variables=None, every=1, last=None):
        """Field snapshots (time, z, y, x) in package convention.
        last : keep only the last `last` snapshots."""
        files = self.field_files()
        files = [(f, t) for f, t in files
                 if (tmin is None or t >= tmin) and (tmax is None or t <= tmax)][::every]
        if last:
            files = files[-last:]
        if not files:
            raise FileNotFoundError(f"{self.label}: no field files '{self.field_prefix}.f*' in {self.out}")
        mesh = None
        allf = self.field_files()
        if allf and "X" in _fields.read_header(allf[0][0])["code"]:
            mesh = allf[0][0]
        fld = _fields.read_fields([f for f, _ in files], variables=variables, rename=self.rename,
                                  mesh=mesh, vertical=self.vertical)
        fld.attrs["label"] = self.label
        fld.attrs["scalar_fields"] = [n for n in self.scalar_fields() if n in fld]
        if "e" in fld and "delta" not in fld:     # SGS TKE models: filter width (elementwise_max)
            dtype = (self.case.les or {}).get("delta_type", "elementwise_max") if self.kind == "neko" \
                else "elementwise_max"
            if dtype == "elementwise_max":
                fld["delta"] = _fields.elementwise_delta(fld)
            else:
                print(f"[nekabl] {self.label}: delta_type '{dtype}' not implemented, no SGS dissipation")
        try:
            fld.attrs["zi"] = self.profile().attrs["zi"]
        except Exception:  # noqa: BLE001  (no statistics: fine)
            pass
        return fld

    def step_series(self, every=1):
        """Per-step series written by the case (Nek5000): ustar, dt, cfl, wall-model
        heat flux and Obukhov length, from diagnostics.dat / wmles.dat. None if absent."""
        if self.kind != "nek5000":
            return None
        from . import nek5000
        return nek5000.read_step_series(self.path, every)

    def n_wall_nodes(self):
        """Number of wall nodes of the wall model = wall elements x (N+1)^2, from the
        first field file (total elements) and the statistics (elements along z)."""
        files = self.field_files()
        N = self.raw.z.attrs.get("N") or self.case.N
        if not files or not N:
            return None
        nel = _fields.read_header(files[0][0])["nel_total"]
        nelz = (self.raw.z.size - 1) // N
        return int(round(nel / nelz)) * (N + 1) ** 2

    def wall_model_series(self, n_wall=None):
        """Per-step output of the wall model as horizontal means (pandas DataFrame
        with t, ustar, q, ts, L, ...): Neko -> 'Wall model diagnostics' of the log
        (sums divided by n_wall, default n_wall_nodes()); Nek5000 -> wmles.dat +
        diagnostics.dat. None if not available."""
        if self.kind == "nek5000":
            s = self.step_series(every=10)
            if s is None:
                return None
            import pandas as pd
            out = pd.DataFrame({"t": s.time.values})
            for k, v in (("wm_q", "q"), ("wm_L", "L")):   # (diagnostics.dat u* is the wall torque)
                if k in s:
                    out[v] = s[k].values
            return out
        lg = self.log()
        if lg is None:
            return None
        from .logfile import wall_model_means
        n = n_wall or self.n_wall_nodes()
        return wall_model_means(lg[0], n) if n else None

    def log(self):
        """(steps, meta) parsed from the log file of the run, or None."""
        from .logfile import read_log
        from .summary import find_log

        f = find_log(self.path, self.out)
        return read_log(f) if f else None
