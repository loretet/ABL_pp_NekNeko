"""
summary.py -- One-page text summary of a Neko run: case parameters, log
(time step, CFL, solver iterations, wall time), statistics and field files,
and the bulk ABL scales of the last statistics windows.

From Python:
    nk.summarize("path/to/case")                # case dir with .case + output/
From a terminal:
    python -m nekabl.summary path/to/case [--log logfile] [--output output] [--save]
    nekabl-summary path/to/case                 # after pip install
"""

import argparse
import glob
import os

import numpy as np

from . import case as _case
from . import fields, logfile, stats


def find_log(*folders):
    """Most recent file that looks like a Neko/Nek5000 log in the folders."""
    pats = ["*.log", "log*", "*.out", "slurm*", "logfile*", "*.err"]
    cands = []
    for f in folders:
        for p in pats:
            cands += [c for c in glob.glob(os.path.join(f, p)) if os.path.isfile(c)]
    good = []
    for c in set(cands):
        try:
            with open(c, errors="ignore") as fh:
                head = fh.read(200000)
            if "Step =" in head or "Step " in head and "DT=" in head:
                good.append(c)
        except OSError:
            pass
    return max(good, key=os.path.getmtime) if good else None


def _files_section(out):
    L = ["---- output files ------------------------------------------"]
    for f in sorted(glob.glob(os.path.join(out, "*.csv"))):
        with open(f) as fh:
            first = fh.readline()
        rows = sum(1 for _ in open(f))
        L.append(f" {os.path.basename(f):<34} {len(first.split(',')) - 2:>3} stats, "
                 f"{rows:>7} rows, {os.path.getsize(f) / 1e6:7.2f} MB")
    series = sorted({os.path.basename(f).split(".f")[0] for f in glob.glob(os.path.join(out, "*.f[0-9]*"))})
    for s in series:
        files = sorted(glob.glob(os.path.join(out, f"{s}.f[0-9]*")))
        h0 = fields.read_header(files[0])
        times = [fields.read_header(f)["time"] for f in files]
        size = sum(os.path.getsize(f) for f in files) / 1e6
        L.append(f" {s + '.f*':<34} {len(files):>3} files, t = {times[0]:.1f} ... {times[-1]:.1f}, "
                 f"vars '{fields.read_header(files[-1])['code']}', {size:.1f} MB")
        if "X" in h0["code"]:
            st = fields._mesh_structure(files[0])
            nz, ny, nx = st["shape"]
            L.append(f"   grid: N = {st['N']}, {h0['nel']} elements, {nx} x {ny} x {nz} unique GLL points, "
                     f"domain {np.ptp(st['x']):.0f} x {np.ptp(st['y']):.0f} x {np.ptp(st['z']):.0f} m")
    return "\n".join(L)


def summarize(case_dir, output="output", log=None, last=None, print_it=True):
    """Build (and print) the summary text. `last` = averaging time (s) used for
    the bulk scales (default: the last half of the statistics)."""
    from .diagnostics import scales, abl_heights
    from .profiles import derive, time_average

    out = os.path.join(case_dir, output) if not os.path.isabs(output) else output
    if not os.path.isdir(out):
        out = case_dir
    parts = ["=" * 62, f" nekabl summary of {os.path.abspath(case_dir)}", "=" * 62]
    case = None
    cpath = _case.find_case(case_dir)
    if cpath:
        case = _case.read_case(cpath)
        parts += ["---- case ---------------------------------------------------", _case.describe(case)]
    log = log or find_log(case_dir, out)
    if log:
        steps, meta = logfile.read_log(log)
        parts += ["---- log ----------------------------------------------------", logfile.describe(steps, meta)]
    else:
        parts.append("---- log: none found (pass --log) ----")
    parts.append(_files_section(out))
    try:
        raw = stats.load_stats(out, case=case, verbose=False)
        t = raw.time.values
        last = last or 0.5 * (t[-1] - (t[0] - raw.window.values[0]))
        prof = derive(time_average(raw, t[-1] - last - 1e-6, None), case=case)
        sc = scales(prof, case)
        zis = abl_heights(prof)
        parts += [f"---- bulk scales (stats average {prof.attrs['t0']:.0f} - {prof.attrs['t1']:.0f} s) ----",
                  f" u*      = {sc['ustar']:.4f} m/s        <w'theta'>_s = {sc['wt_s']:.4f} K m/s",
                  f" theta*  = {sc['thetastar']:.4f} K          L = {sc['L']:.1f} m",
                  f" zi      = {sc['zi']:.1f} m (min heat flux)   w* = {sc['wstar']:.3f} m/s",
                  f" -zi/L   = {sc['zi_L']:.2f}          zi/w* = {sc['t_conv']:.0f} s",
                  " zi by method: " + ", ".join(f"{k} {v:.0f} m" for k, v in zis.items())]
        if "h_ekman" in sc:
            parts.append(f" u*/|f|  = {sc['h_ekman']:.0f} m")
        Ug = case.u_geo if case is not None else None
        top = prof.isel(z=-1)
        parts.append(f" top wind U = {float(top.U):.2f}, V = {float(top.V):.2f} m/s"
                     + (f" (U_g = {Ug})" if Ug else ""))
    except (FileNotFoundError, ValueError, KeyError) as err:
        parts.append(f"---- no bulk scales ({err}) ----")
    text = "\n".join(parts)
    if print_it:
        print(text)
    return text


def main():
    ap = argparse.ArgumentParser(description="Summary of a Neko ABL run")
    ap.add_argument("case_dir", nargs="?", default=".")
    ap.add_argument("--output", default="output", help="output folder (relative to case_dir)")
    ap.add_argument("--log", default=None, help="log file (default: searched)")
    ap.add_argument("--save", action="store_true", help="also write summary.txt in case_dir")
    a = ap.parse_args()
    text = summarize(a.case_dir, a.output, a.log)
    if a.save:
        with open(os.path.join(a.case_dir, "summary.txt"), "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
