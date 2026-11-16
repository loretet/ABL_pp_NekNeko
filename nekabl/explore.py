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
