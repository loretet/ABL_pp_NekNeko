"""
case.py -- Read the Neko case file (JSON) and pull out the physical parameters
an ABL analysis needs (g, theta_0, kappa, z0, surface flux, f, U_g, LES model,
statistics settings, ...).

    case = read_case("shear_convection_abl.case")
    case.T0, case.g, case.z0, case.surface_flux, case.f, case.stats_start_time

Everything is optional: missing entries are None, so the analysis functions
can fall back on their own defaults. Named constants ("constants" block of the
case file, e.g. "g_vector") are resolved automatically.

For Nek5000 there is no JSON case file: build the same object by hand with
`case_from_dict(dict(T0=300, g=9.81, z0=0.1, ...))`.
"""

import glob
import json
import os
from types import SimpleNamespace

import numpy as np

DEFAULTS = dict(
    path=None, json=None, N=None, end_time=None, variable_dt=None, dt=None,
    target_cfl=None, max_dt=None, rho=1.0, nu=None, g=9.81, g_vector=None, T0=None,
    f=None, u_geo=None, wall_model=None, kappa=0.4, z0=None, z0h=None,
    temp_bc_type=None, surface_flux=None, surface_temperature=None, Pr=None,
    h_index=None, les_model=None, les=None, stats=None, stats_start_time=None,
    field_output_value=None, scalars=None, sponge=None,
)


def case_from_dict(d):
    """CaseInfo from a plain dict (useful for Nek5000 or quick tests)."""
    c = dict(DEFAULTS)
    c.update(d)
    return SimpleNamespace(**c)


def _resolve(obj, constants):
    """Replace strings that name a constant by its value, recursively."""
    if isinstance(obj, dict):
        return {k: _resolve(v, constants) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(v, constants) for v in obj]
    if isinstance(obj, str) and obj in constants:
        return constants[obj]
    return obj


def find_case(folder):
    """First *.case file in folder (or its parent)."""
    for f in (folder, os.path.dirname(os.path.abspath(folder))):
        files = sorted(glob.glob(os.path.join(f, "*.case")))
        if files:
            return files[0]
    return None


def read_case(path):
    """Parse a Neko .case file into a CaseInfo namespace (see module doc)."""
    if os.path.isdir(path):
        path = find_case(path)
        if path is None:
            raise FileNotFoundError("no .case file found")
    with open(path) as f:
        js = json.load(f)
    case = js.get("case", js)
    consts = {c["name"]: c["value"] for c in case.get("constants", [])}
    case = _resolve(case, consts)
    c = dict(DEFAULTS)
    c["path"], c["json"] = os.path.abspath(path), case

    num, tim, fl = case.get("numerics", {}), case.get("time", {}), case.get("fluid", {})
    c["N"] = num.get("polynomial_order")
    c["end_time"] = tim.get("end_time", case.get("end_time"))
    c["variable_dt"] = tim.get("variable_timestep", case.get("variable_timestep"))
    c["dt"] = tim.get("timestep", case.get("timestep"))
    c["target_cfl"] = tim.get("target_cfl", case.get("target_cfl"))
    c["max_dt"] = tim.get("max_timestep", case.get("max_timestep"))
    c["field_output_value"] = fl.get("output_value")
    rho = fl.get("rho", 1.0)
    c["rho"] = rho
    if "mu" in fl:
        c["nu"] = fl["mu"] / rho
    elif "Re" in fl:
        c["nu"] = 1.0 / fl["Re"]

    for st in fl.get("source_terms", []):
        t = st.get("type", "")
        if t == "boussinesq":
            c["T0"] = st.get("reference_value")
            gv = st.get("g")
            if isinstance(gv, list):
                c["g_vector"] = gv
                c["g"] = float(np.linalg.norm(gv))
        elif t == "coriolis":
            if "rotation_vector" in st:
                c["f"] = 2.0 * st["rotation_vector"][2]
            elif "f" in st:
                c["f"] = st["f"]
            elif "omega" in st and "phi" in st:
                c["f"] = 2.0 * st["omega"] * np.sin(np.deg2rad(st["phi"]))
            c["u_geo"] = st.get("geostrophic_wind")
        elif t == "sponge":
            c["sponge"] = st

    for bc in fl.get("boundary_conditions", []):
        if bc.get("type") == "wall_model":
            c["wall_model"] = bc.get("model")
            c["kappa"] = bc.get("kappa", 0.4)
            c["z0"] = bc.get("z0")
            c["z0h"] = bc.get("z0h", bc.get("z0"))
            c["Pr"] = bc.get("Pr")
            c["h_index"] = bc.get("h_index")
            c["temp_bc_type"] = bc.get("type_of_temp_bc")
            val = bc.get("bottom_bc_flux_or_temp")
            if c["temp_bc_type"] == "neumann":
                c["surface_flux"] = val
            elif c["temp_bc_type"] == "dirichlet":
                c["surface_temperature"] = val
            if c["g_vector"] is None and isinstance(bc.get("g"), list):
                c["g_vector"] = bc["g"]
                c["g"] = float(np.linalg.norm(bc["g"]))

    c["scalars"] = [s.get("name", "s") for s in case.get("scalars", [])] or (
        [case["scalar"].get("name", "s")] if "scalar" in case else [])

    stats = []
    for sc in case.get("simulation_components", []):
        t = sc.get("type", "")
        if t == "les_model":
            c["les_model"] = sc.get("model")
            c["les"] = sc
            if isinstance(sc.get("T0"), (int, float)):     # Deardorff: own reference temperature
                c["sgs_T0"] = float(sc["T0"])
        if t.endswith("stats"):
            stats.append(dict(type=t, field=sc.get("field"), start_time=sc.get("start_time", 0.0),
                              avg_direction=sc.get("avg_direction"),
                              set_of_stats=sc.get("set_of_stats", "full" if t in ("fluid_stats", "scalar_stats") else None),
                              output_value=sc.get("output_value"), compute_value=sc.get("compute_value")))
    c["stats"] = stats
    fs = [s for s in stats if s["type"] == "fluid_stats"] or stats
    c["stats_start_time"] = fs[0]["start_time"] if fs else None
    return SimpleNamespace(**c)


def describe(case):
    """Short human-readable text of the main physical parameters."""
    L = []
    add = L.append
    add(f"case file      : {case.path}")
    add(f"polynomial ord.: N = {case.N}")
    add(f"end time       : {case.end_time} s   (variable dt: {case.variable_dt}, "
        f"target CFL {case.target_cfl}, max dt {case.max_dt})")
    add(f"nu             : {case.nu}")
    add(f"Boussinesq     : theta_0 = {case.T0} K, g = {case.g} m/s2")
    if case.f is not None:
        add(f"Coriolis       : f = {case.f:.3e} 1/s, U_g = {case.u_geo}")
    if case.wall_model:
        hs = (f"sampling height h = {case.h} m" if getattr(case, "h", None)
              else f"h_index = {case.h_index}")
        add(f"wall model     : {case.wall_model}, kappa = {case.kappa}, z0 = {case.z0}, "
            f"z0h = {case.z0h}, {hs}")
        if case.surface_flux is not None:
            add(f"surface flux   : w'theta'_s = {case.surface_flux} K m/s (Neumann)")
        if case.surface_temperature is not None:
            add(f"surface temp.  : theta_s = {case.surface_temperature} K (Dirichlet)")
    if case.les_model:
        extra = {k: v for k, v in (case.les or {}).items() if k not in ("type", "model", "g")}
        if getattr(case, "sgs_T0", None):
            extra.update(T0=case.sgs_T0, g=getattr(case, "sgs_g", None))
        add(f"LES model      : {case.les_model} {extra}")
    stats = case.stats or []
    if isinstance(stats, dict):          # Nek5000 [_STAT] section
        add(f"stats          : KTH statistics module, {stats}")
    for s in ([] if isinstance(stats, dict) else stats):
        add(f"stats          : {s['type']:<17} field={s['field']}, set={s['set_of_stats']}, "
            f"avg={s['avg_direction']}, start={s['start_time']}, every {s['compute_value']}, "
            f"write {s['output_value']}")
    return "\n".join(L)
