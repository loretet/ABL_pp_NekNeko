"""
sem.py -- Small spectral-element (GLL) helpers used everywhere in the package.

Neko and Nek5000 store data on Gauss-Lobatto-Legendre (GLL) points inside each
element. For a Cartesian box mesh every axis is simply a sequence of elements,
each holding N+1 GLL points (N = polynomial order) with shared end points.

The `Axis` class describes one such axis and gives you, as plain matrices:
    * quadrature weights        -> exact integrals / averages
    * derivative matrix         -> spectrally accurate d/dz (or d/dx, d/dy)
    * interpolation matrix      -> values at any coordinate (e.g. a given height)

If the coordinates do not follow a GLL structure (e.g. data already
interpolated on a uniform grid) a finite-difference fallback with the same
interface is returned by `make_axis`, so the rest of the code never has to care.
"""

import numpy as np
from numpy.polynomial import legendre as _leg


# ----------------------------------------------------------------------------
# 1D GLL building blocks on the reference element [-1, 1]
# ----------------------------------------------------------------------------
def gll(N):
    """GLL nodes and weights for polynomial order N (N+1 points)."""
    cN = np.zeros(N + 1)
    cN[N] = 1.0
    inner = _leg.legroots(_leg.legder(cN)) if N > 1 else np.array([])
    xi = np.concatenate(([-1.0], np.sort(np.real(inner)), [1.0]))
    PN = _leg.legval(xi, cN)
    w = 2.0 / (N * (N + 1) * PN**2)
    return xi, w


def lagrange_matrix(xi, xt):
    """Matrix L (len(xt), len(xi)) such that f(xt) = L @ f(xi)."""
    xi = np.asarray(xi, float)
    xt = np.atleast_1d(np.asarray(xt, float))
    L = np.ones((xt.size, xi.size))
    for j in range(xi.size):
        for m in range(xi.size):
            if m != j:
                L[:, j] *= (xt - xi[m]) / (xi[j] - xi[m])
    return L


def derivative_matrix(N):
    """GLL collocation derivative matrix on [-1, 1]: (df/dxi)_i = D_ij f_j."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Global axes
# ----------------------------------------------------------------------------
class Axis:
    """A 1D spectral-element axis (unique GLL coordinates, shared end points).

    Parameters
    ----------
    coord : 1D array, length nel*N + 1, increasing.
    N     : polynomial order.
    """

    kind = "sem"

    def __init__(self, coord, N):
        self.coord = np.asarray(coord, float)
        self.N = int(N)
        self.n = self.coord.size
        self.nel = (self.n - 1) // self.N
        self.edges = self.coord[:: self.N]
        self.h = np.diff(self.edges)
        self.xi, self.wxi = gll(self.N)

    # -- integration --------------------------------------------------------
    def weights(self):
        """Global quadrature weights (sum = axis length)."""
        w = np.zeros(self.n)
        for e in range(self.nel):
            w[e * self.N:(e + 1) * self.N + 1] += self.wxi * self.h[e] / 2.0
        return w

    # -- differentiation ----------------------------------------------------
    def derivative(self):
        """Global derivative matrix (n, n); interface values are averaged
        between the two neighbouring elements (as Neko's gather-scatter)."""
        Dref = derivative_matrix(self.N)
        D = np.zeros((self.n, self.n))
        count = np.zeros(self.n)
        for e in range(self.nel):
            sl = slice(e * self.N, (e + 1) * self.N + 1)
            D[sl, sl] += Dref * 2.0 / self.h[e]
            count[sl] += 1
        return D / count[:, None]

    def diff(self, arr, axis=-1):
        """Derivative of a numpy array along `axis`, element by element
        (O(n N) instead of the O(n^2) dense matrix; same result)."""
        a = np.moveaxis(np.asarray(arr, float), axis, -1)
        Dref = derivative_matrix(self.N)
        win = np.lib.stride_tricks.sliding_window_view(a, self.N + 1, axis=-1)[..., :: self.N, :]
        de = np.einsum("ij,...ej->...ei", Dref, win) * (2.0 / self.h)[:, None]
        out = np.zeros_like(a)
        cnt = np.zeros(self.n)
        for e in range(self.nel):
            sl = slice(e * self.N, (e + 1) * self.N + 1)
            out[..., sl] += de[..., e, :]
            cnt[sl] += 1
        return np.moveaxis(out / cnt, -1, axis)

    # -- interpolation ------------------------------------------------------
    def interpolation(self, xt):
        """Matrix (len(xt), n) giving the spectral interpolant at points xt."""
        xt = np.atleast_1d(np.asarray(xt, float))
        M = np.zeros((xt.size, self.n))
        e = np.clip(np.searchsorted(self.edges, xt, side="right") - 1, 0, self.nel - 1)
        for k in np.unique(e):
            sel = e == k
            xloc = 2.0 * (xt[sel] - self.edges[k]) / self.h[k] - 1.0
            M[np.ix_(sel, np.arange(k * self.N, (k + 1) * self.N + 1))] = lagrange_matrix(self.xi, xloc)
        return M
