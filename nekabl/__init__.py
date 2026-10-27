"""
nekabl -- exploratory analysis of atmospheric boundary layer LES from Neko and
Nek5000 (z is always the vertical direction).

Modules (each can be used on its own; the most used functions are also
available directly as nk.<function>):

    entrainment    inversion, entrainment fluxes / rates of momentum, heat, scalars
    case           read the Neko .case file (physical parameters)
    logfile        parse the run log (dt, CFL, iterations, timings)
    stats          read statistics (CSV, 3D stats fld, Nek5000 avg files)
    fields         read field files (fld) on box meshes, planes, averages
    sem            GLL helpers (exact derivatives, integrals, interpolation)
    profiles       raw statistics -> mean profiles (+ profile plots)
    diagnostics    u*, L, zi, w*, MOST / wall-model / SGS / resolution checks
    turbulence     budgets, eddy viscosity, spectra, autocorrelation, quadrants
    visualization  horizontal and vertical slices
    most           Monin-Obukhov functions (same as the Neko wall model)
    plotstyle      common figure setup (font size, DPI, LaTeX, colours)
    summary        one-page text summary of a run
    run            Run: one object per simulation folder (Neko or Nek5000)
    nek5000        Nek5000 interface (y-up rotation, KTH statistics, .par)
    explore        comprehensive exploration from the command line (nekabl-explore)
"""
__version__ = "0.5.0"
