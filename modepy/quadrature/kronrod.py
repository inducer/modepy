"""
.. autoclass:: KronrodGaussQuadrature
.. autoclass:: KronrodQuadrature
.. autofunction:: make_kronrod_quadrature


References
----------
.. [Laurie1997] Dirk P. Laurie, "Calculation of Gauss-Kronrod quadrature rules,"
    Mathematics of Computation, vol. 66, no. 219, pp. 1133-1145 (1997).
    `DOI <https://doi.org/10.1090/S0025-5718-97-00861-2>`__
"""

from __future__ import annotations


__copyright__ = """
Copyright (C) 2013 Steven G. Johnson
Copyright (C) 2026 University of Illinois Board of Trustees

Ported by Claude from
https://github.com/stevengj/julia/blob/949421f4eedbdf0493a6080a2ecd4f862f051c68/base/quadgk.jl
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import heapq
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, overload

import numpy as np

from modepy import Quadrature


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import optype.numpy as onp

    from modepy.typing import ArrayF


class ScalarOrArrayIntegrand(Protocol):
    @overload
    def __call__(self,
                 /, x: onp.Array1D[np.floating],
             ) -> onp.Array1D[np.complexfloating]: ...

    @overload
    def __call__(self, /, x: float, ) -> complex: ...


# ---------------------------------------------------------------------------
# Segment  (priority-queue element, ordered by error descending)
# ---------------------------------------------------------------------------

@dataclass(order=False, frozen=True)
class Segment:
    a: complex
    b: complex
    approx_integral: complex
    error_est: float

    # Max-heap via negated error (Python's heapq is a min-heap)
    def __lt__(self, other: Segment) -> bool:
        # reversed so heapq gives the largest error
        return self.error_est > other.error_est

    def __le__(self, other: Segment) -> bool:
        return self.error_est >= other.error_est


# ---------------------------------------------------------------------------
# Eigen-utilities for Kronrod/Gauss rule construction
# ---------------------------------------------------------------------------

def _eigpoly(b: np.ndarray, z, m: int | None = None):
    """
    Given the off-diagonal entries b of a symmetric tridiagonal matrix H
    (diagonal = 0, H[i-1,i] = b[i-1]), compute p(z) = det(z*I - H) and
    p'(z) via the three-term recurrence.

    Returns (p, p_prime).
    """
    if m is None:
        m = len(b) + 1
    d1 = z
    d1d = type(z)(1)   # d1 derivative
    d2 = type(z)(1)
    d2d = type(z)(0)
    for i in range(2, m + 1):
        b2 = b[i - 2] ** 2
        d = z * d1 - b2 * d2
        dd = d1 + z * d1d - b2 * d2d
        d2, d1 = d1, d
        d2d, d1d = d1d, dd
    return d1, d1d


def _eignewt(b: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    Compute the n smallest eigenvalues of the symmetric tridiagonal matrix
    defined by off-diagonal entries b (length m-1), using Newton's method
    seeded from numpy's eigvalsh.
    """
    # Build the tridiagonal matrix for the seed step
    h_mat = np.zeros((m, m), dtype=float)
    for i in range(1, m):
        h_mat[i - 1, i] = h_mat[i, i - 1] = float(b[i - 1])
    lam0 = np.sort(np.linalg.eigvalsh(h_mat))

    lam = np.array(lam0[:n], dtype=b.dtype)
    for i in range(n):
        for _ in range(1000):
            p, pd = _eigpoly(b, lam[i], m)
            lam_old = lam[i]
            if pd == 0:
                break
            lam[i] = lam_old - p / pd
            if abs(lam[i] - lam_old) < 10 * np.finfo(float).eps * abs(lam[i]):
                break
        # One final Newton step
        p, pd = _eigpoly(b, lam[i], m)
        if pd != 0:
            lam[i] -= p / pd
    return lam


def _eigvec1(b: np.ndarray, z, m: int | None = None) -> np.ndarray:
    """
    Given eigenvalue z and the tridiagonal matrix defined by b, return the
    corresponding normalised eigenvector.  Exploits the fact that the first
    component is nonzero (it encodes a quadrature weight), so we set v[0]=1
    and solve forward.
    """
    if m is None:
        m = len(b) + 1
    v = np.empty(m, dtype=type(z))
    v[0] = type(z)(1)
    if m > 1:
        s = v[0] ** 2
        v[1] = z * v[0] / b[0]
        s += v[1] ** 2
        for i in range(2, m):
            v[i] = -(b[i - 2] * v[i - 2] - z * v[i - 1]) / b[i - 1]
            s += v[i] ** 2
        v /= np.sqrt(float(s))
    return v


# ---------------------------------------------------------------------------
# Public: gauss — N-point Gauss-Legendre rule on [-1, 1]
# ---------------------------------------------------------------------------

def gauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (x, w) — N Gauss-Legendre quadrature points and weights on [-1,1].

    dot(f(x), w) approximates integral_{-1}^{1} f(t) dt.
    """
    if n < 1:
        raise ValueError("Gauss rules require positive order")
    b = np.array([n / np.sqrt(4 * n * n - 1.0) for n in range(1, n)], dtype=float)
    x = _eignewt(b, n, n)
    w = np.array([2.0 * _eigvec1(b, x[i])[0] ** 2 for i in range(n)])
    return x, w


@lru_cache
def kronrod(n: int) -> tuple[
            onp.Array1D[np.floating],
            onp.Array1D[np.floating],
            onp.Array1D[np.floating]
        ]:
    """
    Return (x, w, gw) for the (2n+1)-point Gauss-Kronrod rule on [-1,1].

    Only the n+1 points with x <= 0 are returned (the rule is symmetric).
    The node at zero is guaranteed to be the last node in *x*.

    x  — Kronrod quadrature points (length n+1)
    w  — Kronrod weights            (length n+1)
    gw — embedded n-point Gauss weights for x[1::2] (length n//2 + (1 if n%2 else 0))

    Based on [Laurie1997]_, Appendix A, specialised to the unit weight on [-1,1].
    """
    if n < 1:
        raise ValueError("Kronrod rules require positive order")

    b = np.zeros(2 * n + 1, dtype=float)
    b[0] = 2.0
    for j in range(1, (3 * n + 1) // 2 + 1):
        b[j] = j * j / (4.0 * j * j - 1.0)

    half = n // 2 + 2
    s = np.zeros(half, dtype=float)
    t = np.zeros(half, dtype=float)
    t[1] = b[n + 1]

    for m in range(n - 1):
        u = 0.0
        for k in range((m + 1) // 2, -1, -1):
            ell = m - k + 1
            k1 = k + n + 1          # 0-based: Julia's k+n+2 -> k+n+1
            u += b[k1] * s[k] - b[ell - 1] * s[k + 1]
            s[k + 1] = u
        s, t = t.copy(), s.copy()

    for j in range(n // 2, -1, -1):
        s[j + 1] = s[j]

    for m in range(n - 1, 2 * n - 2):
        u = 0.0
        for k in range(m + 1 - n, (m - 1) // 2 + 1):
            ell = m - k + 1
            j = n - ell
            k1 = k + n + 1
            u -= b[k1] * s[j + 1] - b[ell - 1] * s[j + 2]
            s[j + 1] = u
        k = (m + 1) // 2
        if 2 * k != m:
            j = n - (m - k + 2)
            b[k + n + 1] = s[j + 1] / s[j + 2]
        s, t = t.copy(), s.copy()

    for j in range(2 * n):
        b[j] = np.sqrt(b[j + 1])

    # Negative quadrature points
    x = _eignewt(b, 2 * n + 1, n + 1)

    # Kronrod weights
    w = np.array([2.0 * _eigvec1(b, x[i], 2 * n + 1)[0] ** 2 for i in range(n + 1)])

    # Embedded Gauss weights (even-indexed points: x[1], x[3], ...)
    bg = np.array([j / np.sqrt(4.0 * j * j - 1.0) for j in range(1, n)], dtype=float)
    # even-indexed in 0-based: indices 1, 3, 5, ... i.e. range(1, n+1, 2)
    gw = np.array([2.0 * _eigvec1(bg, x[i], n)[0] ** 2 for i in range(1, n + 1, 2)])

    # A feeble attempt at making mutable caching safe
    x.setflags(write=False)
    w.setflags(write=False)
    gw.setflags(write=False)

    return x, w, gw


class KronrodGaussQuadrature(Quadrature):
    pass


@dataclass(frozen=True)
class KronrodQuadrature:
    """Uses a computation following [Laurie1997]_.

    .. autoattribute:: nodes
    .. autoattribute:: weights
    .. autoattribute:: weights_at_gauss_nodes
    .. autoattribute:: gauss_quadrature
    .. autoattribute:: exact_to

    .. automethod:: __call__
    """
    nodes: ArrayF
    """
    an array of shape *(d, nnodes)*, where *d* is the dimension of the quadrature rule.
    """

    weights: ArrayF
    """
    an array of length *nnodes*.
    """

    weights_at_gauss_nodes: ArrayF

    exact_to: int
    gauss_quadrature: KronrodGaussQuadrature

    def __call__(
                self,
                f: Callable[[ArrayF], ArrayF]
            ) -> ArrayF | np.floating:
        """Evaluate the Kronrod estimate for the integral.

        .. note::

            This will (re-)evaluate *f* at both the Gauss and the Kronrod nodes,
            likely negating any cost advantage of Kronrod.
            As a result, it is recommended to reimplement the evaluation
            logic, following the code of this function, while
            reusing the function values from the Gauss evaluation.
        """

        return (
            f(self.gauss_quadrature.nodes) @ self.weights_at_gauss_nodes
            +
            f(self.nodes) @ self.weights
        )

    @property
    def dim(self) -> int:
        """Dimension of the space on which the quadrature rule applies."""
        return 1 if self.nodes.ndim == 1 else self.nodes.shape[0]


def make_kronrod_quadrature(n: int) -> KronrodQuadrature:
    x, w, gw = kronrod(n)

    gauss_neg = x[1::2]   # non-positive Gauss nodes  (includes 0 iff n is odd)
    w_gauss = w[1::2]   # Kronrod weights at Gauss nodes
    kron_neg  = x[::2]    # non-positive Kronrod-only  (includes 0 iff n is even)
    w_kron    = w[::2]    # Kronrod weights at the Kronrod-only positions

    if n % 2 == 1:
        # n odd: centre belongs to Gauss; all kron_neg are strictly negative
        gauss_x = np.concatenate((gauss_neg[:-1], [0.0], -gauss_neg[-2::-1]))
        gauss_w = np.concatenate((gw[:-1], [gw[-1]], gw[-2::-1]))
        kron_x  = np.concatenate((kron_neg, -kron_neg[::-1]))
        kron_w  = np.concatenate((w_kron, w_kron[::-1]))
        kron_gauss_w = np.concatenate((w_gauss[:-1], [w[-1]], w_gauss[-2::-1]))
    else:
        # n even: centre belongs to Kronrod-only; all gauss_neg are strictly negative
        gauss_x = np.concatenate((gauss_neg, -gauss_neg[::-1]))
        gauss_w = np.concatenate((gw, gw[::-1]))
        kron_x  = np.concatenate((kron_neg[:-1], [0.0], -kron_neg[-2::-1]))
        kron_w  = np.concatenate((w_kron[:-1], [w_kron[-1]], w_kron[-2::-1]))
        kron_gauss_w = np.concatenate((w_gauss, w_gauss[::-1]))

    gauss_rule = KronrodGaussQuadrature(
        nodes=gauss_x,
        weights=gauss_w,
        exact_to=2*n-1,
    )
    return KronrodQuadrature(
        nodes=kron_x,
        weights=kron_w,
        weights_at_gauss_nodes=kron_gauss_w,
        exact_to=3*n+1,
        gauss_quadrature=gauss_rule,
    )


def _evalrule(
            f: ScalarOrArrayIntegrand,
            a: float,
            b: float,
            x: np.ndarray,
            w: np.ndarray,
            gw: np.ndarray) -> Segment:
    """
    Apply the Gauss-Kronrod rule (x, w, gw) to f on [a, b].
    Returns a Segment with the Kronrod estimate I and error estimate E.
    """
    s = 0.5 * (b - a)
    nx = len(x)
    ngw = len(gw)
    n1 = 1 - (nx & 1)   # 0 if even-length x, 1 if odd-length x

    i_k = 0.0
    i_g = 0.0

    # Julia loop: for i = 1:length(gw)-n1  (1-based)
    #   fg uses x[2i],   w[2i]   (1-based) -> x[2i-1],   w[2i-1]   (0-based)
    #   fk uses x[2i-1], w[2i-1] (1-based) -> x[2i-2],   w[2i-2]   (0-based)
    for i in range(1, ngw - n1 + 1):          # i is 1-based to match Julia
        fg = f(a + (1 + x[2*i - 1]) * s) + f(a + (1 - x[2*i - 1]) * s)
        fk = f(a + (1 + x[2*i - 2]) * s) + f(a + (1 - x[2*i - 2]) * s)
        i_g += fg * gw[i - 1]
        i_k += fg * w[2*i - 1] + fk * w[2*i - 2]

    if n1 == 0:   # even-order: centre point (x==0) appears only in Kronrod
        i_k += f(a + s) * w[-1]
    else:         # odd-order: centre point also in embedded Gauss rule
        f0 = f(a + s)
        i_g += f0 * gw[-1]
        # x[-2] is the last non-centre Kronrod point (0-based index nx-2)
        i_k += f0 * w[-1] + (f(a + (1 + x[-2]) * s) + f(a + (1 - x[-2]) * s)) * w[-2]

    i_k *= s
    i_g *= s
    err_est = abs(i_k - i_g)
    if np.isnan(err_est) or np.isinf(err_est):
        raise ValueError(
            f"quadgk: integrand returned NaN or Inf on [{a}, {b}]; "
            "check for singularities."
        )
    return Segment(a, b, i_k, err_est)


# ---------------------------------------------------------------------------
# Internal: coordinate transformations for infinite intervals
# ---------------------------------------------------------------------------

def _transform_both_inf(f, s):
    """x = t / (1 - t^2), maps (-1,1) -> (-inf, inf)."""
    def g(t):
        t2 = t * t
        den = 1.0 / (1.0 - t2)
        return f(t * den) * (1.0 + t2) * den * den

    def _map(x):
        if np.isinf(x):
            return np.copysign(1.0, x)
        return 2.0 * x / (1.0 + np.hypot(1.0, 2.0 * x))

    return g, [_map(xi) for xi in s]


def _transform_semi_inf_neg(f, s0, s):
    """x = s0 - t/(1-t), maps (0,1) -> (-inf, s0)."""
    def g(t):
        den = 1.0 / (1.0 - t)
        return f(s0 - t * den) * den * den

    def _map(x):
        if np.isinf(x):
            return 1.0   # -inf maps to t=1, the far end of (0,1)
        diff = s0 - x
        if diff == 0.0:  # noqa: RUF069
            return 0.0
        return 1.0 / (1.0 + 1.0 / diff)

    mapped = [_map(xi) for xi in s]
    mapped.reverse()
    return g, mapped


def _transform_semi_inf_pos(f, s0, s):
    """x = s0 + t/(1-t), maps (0,1) -> (s0, +inf)."""
    def g(t):
        den = 1.0 / (1.0 - t)
        return f(s0 + t * den) * den * den

    def _map(x):
        if np.isinf(x):
            return 1.0
        diff = x - s0
        if diff == 0.0:  # noqa: RUF069
            return 0.0
        return 1.0 / (1.0 + 1.0 / diff)

    return g, [_map(xi) for xi in s]


# ---------------------------------------------------------------------------
# Internal: do_quadgk
# ---------------------------------------------------------------------------

def _do_quadgk(
            f: ScalarOrArrayIntegrand,
            s: Sequence[float],
            n: int,
            abstol: float,
            reltol: float,
            maxevals: int,
        ) -> tuple[complex, float]:
    """
    Integrate f over the union of open intervals defined by the breakpoints s,
    using h-adaptive Gauss-Kronrod quadrature.
    """
    # Handle infinite / semi-infinite intervals
    real_endpoints = all(isinstance(si, (int, float)) for si in s)
    if real_endpoints:
        s1, s2 = s[0], s[-1]
        inf1, inf2 = np.isinf(s1), np.isinf(s2)
        if inf1 or inf2:
            if inf1 and inf2:
                g, sm = _transform_both_inf(f, s)
            elif inf1:
                g, sm = _transform_semi_inf_neg(f, s2, s)
            else:
                g, sm = _transform_semi_inf_pos(f, s1, s)
            return _do_quadgk(g, sm, n, abstol, reltol, maxevals)

    x, w, gw = kronrod(n)

    # Build initial segment heap (max-heap by error via Segment.__lt__)
    heap: list[Segment] = []
    for i in range(len(s) - 1):
        seg = _evalrule(f, s[i], s[i + 1], x, w, gw)
        heapq.heappush(heap, seg)

    numevals = (2 * n + 1) * len(heap)
    approx_int = sum(seg.approx_integral for seg in heap)
    error_est = sum(seg.error_est for seg in heap)

    # h-adaptive refinement: always split the segment with largest error
    while (abstol < error_est
            and reltol * abs(approx_int) < error_est
            and numevals < maxevals):
        worst = heapq.heappop(heap)
        mid = (worst.a + worst.b) * 0.5
        s1 = _evalrule(f, worst.a, mid, x, w, gw)
        s2 = _evalrule(f, mid, worst.b, x, w, gw)
        heapq.heappush(heap, s1)
        heapq.heappush(heap, s2)
        approx_int = ((approx_int - worst.approx_integral)
            + s1.approx_integral + s2.approx_integral)
        error_est = (error_est - worst.error_est) + s1.error_est + s2.error_est
        numevals += 4 * n + 2

    # Final re-sum (guards against accumulated roundoff)
    approx_int = sum(seg.approx_integral for seg in heap)
    error_est = sum(seg.error_est for seg in heap)
    return approx_int, error_est


# ---------------------------------------------------------------------------
# Public: quadgk
# ---------------------------------------------------------------------------

def quadgk(
    f: ScalarOrArrayIntegrand,
    *pts: float,
    abstol: float = 0.0,
    reltol: float = 1.49e-8,   # ~sqrt(eps) for float64; matches Julia's eps(T)*100
    maxevals: int = 10_000_000,
    order: int = 7,
) -> tuple[complex, float]:
    """
    Adaptive Gauss-Kronrod quadrature of f over a sequence of intervals.

    Parameters
    ----------
    f        : callable — the integrand
    *pts     : two or more breakpoints a, b[, c, d, ...].
               Infinite endpoints (±np.inf) are supported.
    abstol   : absolute error tolerance (default 0)
    reltol   : relative error tolerance (default ~1.49e-8)
    maxevals : approximate maximum number of f evaluations (default 10^7)
    order    : Kronrod order n; rule has 2n+1 points (default 7)

    Returns
    -------
    (I, E) — estimated integral and error estimate

    Examples
    --------
    >>> I, E = quadgk(lambda x: x**2, 0, 1)
    >>> abs(I - 1/3) < 1e-10
    np.True_

    >>> I, E = quadgk(lambda x: np.exp(-x**2), -np.inf, np.inf)
    >>> abs(I - np.sqrt(np.pi)) < 1e-10
    np.True_

    >>> I, E = quadgk(lambda x: 1/x**2, 1, np.inf)
    >>> abs(I - 1.0) < 1e-10
    np.True_
    """
    if len(pts) < 2:
        raise ValueError("quadgk requires at least two breakpoints")

    s = list(pts)
    return _do_quadgk(f, s, order, abstol, reltol, maxevals)


def test_zero_node_is_last():
    for n in [2, 3, 5, 7, 8, 12]:
        x, _w, _gw = kronrod(n)
        assert x[-1] == 0


def check(desc: str, got: complex, expected: complex, tol: float = 1e-10):
    err = abs(got - expected)
    assert err < tol, f"{desc}: got {got:.15g}, expected {expected:.15g}, err {err:.3e}"


def test_basic_polynomial():
    approx_int, err_est = quadgk(lambda x: x**2, 0, 1)
    assert err_est < 1e-10
    check("integral x^2 from 0 to 1", approx_int, 1/3)


def test_gauss_inf_interval():
    # Gaussian integral on (-inf, inf)
    approx_int, err_est = quadgk(lambda x: np.exp(-x**2), -np.inf, np.inf)
    assert err_est < 1e-8
    check("integral exp(-x^2) on (-inf,inf)", approx_int, np.sqrt(np.pi))


def test_semi_pos_inf():
    # Semi-infinite: 1/x^2 from 1 to inf
    approx_int, err_est = quadgk(lambda x: 1.0/x**2, 1, np.inf)
    assert err_est < 1e-10
    check("integral 1/x^2 from 1 to inf", approx_int, 1.0)


def test_semi_pos_neg():
    # Semi-infinite negative: e^x from -inf to 0
    approx_int, err_est = quadgk(lambda x: np.exp(x), -np.inf, 0)
    assert err_est < 1e-10
    check("integral e^x from -inf to 0", approx_int, 1.0)


def test_multi_break():
    # Multiple breakpoints
    approx_int, err_est = quadgk(lambda x: abs(x), -1, 0, 1)
    assert err_est < 1e-10
    check("integral |x| from -1 to 1 (with breakpoint)", approx_int, 1.0)


def test_oscillatory():
    # Oscillatory
    approx_int, err_est = quadgk(lambda x: np.sin(x), 0, np.pi)
    assert err_est < 1e-10
    check("integral sin(x) from 0 to pi", approx_int, 2.0)


def test_kronrod_exactness() -> None:
    for n in range(1, 13 + 1):
        krule = make_kronrod_quadrature(n)
        for deg in range(krule.exact_to + 1):
            def f(x: ArrayF) -> ArrayF:
                return np.cos(deg * np.arccos(x))  # noqa: B023

            ik_f = krule(f)
            ig_f = krule.gauss_quadrature(f)

            i_f_true = ((-1)**deg + 1)/(1-deg**2) if deg != 1 else 0

            if deg <= krule.gauss_quadrature.exact_to:
                err = abs(ig_f - i_f_true)
                assert err < 4.0e-15, (n, deg, err, ig_f, i_f_true)

            err = abs(ik_f - i_f_true)
            assert err < 4.0e-15, (n, deg, err, ik_f, i_f_true)
