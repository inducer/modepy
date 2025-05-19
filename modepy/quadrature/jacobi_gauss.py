"""
.. currentmodule:: modepy

.. autoclass:: JacobiGaussQuadrature
    :members:
    :show-inheritance:

.. autoclass:: LegendreGaussQuadrature
    :show-inheritance:

.. autoclass:: ChebyshevGaussQuadrature
    :show-inheritance:

.. autoclass:: GaussGegenbauerQuadrature
    :show-inheritance:

.. currentmodule:: modepy.quadrature.jacobi_gauss

.. autofunction:: jacobi_gauss_lobatto_nodes

.. autofunction:: legendre_gauss_lobatto_nodes

.. currentmodule:: modepy

.. autoclass:: JacobiGaussLobattoQuadrature
.. autoclass:: LegendreGaussLobattoQuadrature
"""
from __future__ import annotations


__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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

import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray

from modepy.quadrature import Quadrature


class JacobiGaussQuadrature(Quadrature):
    r"""A Gauss quadrature of order *N* associated with the Jacobi polynomials.

    The quadrature rule can be used for weighted integrals of the form

    .. math::

        I[f] = \int_{-1}^1 f(x) (1 - x)^\alpha (1 + x)^\beta\, \mathrm{d}x,

    where :math:`\alpha, \beta > -1`. The quadrature rule is exact
    up to degree :math:`2N + 1`. The quadrature has *N+1* nodes.

    .. automethod:: __init__
    """

    def __init__(self,
            alpha: float, beta: float, N: int,                  # noqa: N803
            backend: str | None = None,
            force_dim_axis: bool = False) -> None:
        r"""
        :arg backend: Either ``"builtin"`` or ``"scipy"``. When the
            ``"builtin"`` backend is in use, there is an additional
            requirement that :math:`\alpha + \beta \ne -1`, with the exception
            of the Chebyshev quadrature :math:`\alpha = \beta = -1/2`. The
            ``"scipy"`` backend has no such restriction.
        """
        if not force_dim_axis:
            from warnings import warn
            warn("setting 'force_dim_axis' to 'False' is deprecated and "
                    "makes 1d rules inconsistent with higher dimensions. "
                    "This option will go away in 2022",
                    DeprecationWarning, stacklevel=2)

        if backend is None:
            backend = "builtin"

        if backend == "builtin":
            x, w = self.compute_weights_and_nodes(N, alpha, beta)
        elif backend == "scipy":
            try:
                from scipy.special import roots_jacobi
            except ImportError:
                # NOTE: deprecated in scipy >=1.8.0
                from scipy.special.orthogonal import roots_jacobi

            x, w = roots_jacobi(N + 1, alpha, beta)
        else:
            raise NotImplementedError(f"Unsupported backend: {backend}")

        if force_dim_axis:
            x = x.reshape(1, -1)

        exact_to = 2*N + 1
        super().__init__(x, w, exact_to=exact_to)

        self.alpha: float = alpha
        """Power of :math:`(1 - x)` term in Jacobi quadrature."""
        self.beta: float = beta
        """Power of :math:`(1 + x)` term in Jacobi quadrature."""

    @staticmethod
    def compute_weights_and_nodes(
            N: int, alpha: float, beta: float,  # noqa: N803
            ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        :arg N: order of the Gauss quadrature (the order of exactly
            integrated polynomials is :math:`2 N + 1`).
        :arg alpha: power of :math:`1 - x` in the Jacobi polynomial weight.
        :arg beta: power of :math:`1 + x` in the Jacobi polynomial weight.

        :return: a tuple ``(nodes, weights)`` of quadrature notes and weights.
        """

        # FIXME: This could stand to be upgraded to the Rokhlin algorithm.

        # follows
        # Gene H. Golub, John H. Welsch, Calculation of Gauss Quadrature Rules,
        # Mathematics of Computation, Vol. 23, No. 106 (Apr., 1969), pp. 221-230
        # http://dx.doi.org/10.2307/2004418

        # see also doc/hedge-notes.tm for correspondence with the Jacobi
        # recursion from Hesthaven/Warburton's book

        apb = alpha+beta

        if abs(alpha + 0.5) < 1.0e-14 and abs(beta + 0.5) < 1.0e-14:
            # NOTE: these are hardcoded for two reasons:
            #   * the algorithm below doesn't work for alpha = beta = -0.5
            #   * and, well, we know them explicitly so why not.
            return (
                    np.cos(np.pi / (2 * (N+1)) * (2*np.arange(N+1, 0, -1) - 1)),
                    np.full(N+1, np.pi / (N + 1))
                    )
        elif abs(apb + 1.0) < 1.0e-14:
            raise ValueError("cannot generate Gauss-Jacobi quadrature rules for "
                    f"alpha + beta = 1: ({alpha}, {beta})")

        # see Appendix A of Hesthaven/Warburton for these formulas
        def a(n: int) -> np.floating:
            return 2 / (2*n + apb) * np.sqrt(
                (n * (n+apb) * (n+alpha) * (n+beta))
                / ((2*n + apb - 1) * (2*n + apb + 1))
                )

        def b(n: int) -> float:
            if n == 0:
                return -(alpha-beta) / (apb+2)
            else:
                return -(alpha**2 - beta**2) / ((2*n + apb) * (2*n + apb + 2))

        T = np.zeros((N + 1, N + 1))    # noqa: N806
        current_a = 0.0

        for n in range(N + 1):
            T[n, n] = b(n)
            if n > 0:
                T[n, n-1] = current_a
            if n < N:
                next_a = a(n + 1)
                T[n, n+1] = next_a
                current_a = next_a

        assert la.norm(T - T.T) < 1.0e-12
        eigval, eigvec = la.eigh(T)

        assert la.norm(np.dot(T, eigvec) - np.dot(eigvec, np.diag(eigval))) < 1e-12

        from functools import partial

        from modepy.modes import jacobi
        p0 = partial(jacobi, alpha, beta, 0)  # that's a constant, sure
        nodes = eigval
        weights = np.array(
                [eigvec[0, i]**2 / p0(nodes[i])**2 for i in range(N + 1)])

        return nodes, weights


class LegendreGaussQuadrature(JacobiGaussQuadrature):
    r"""A Gauss quadrature rule with weight :math:`1` and *N+1* nodes.

    Corresponds to a Gauss-Jacobi quadrature rule with
    :math:`\alpha = \beta = 0`.
    """

    def __init__(self,
                 N: int, backend:  # noqa: N803
                 str | None = None,
                 force_dim_axis: bool = False) -> None:
        super().__init__(
                0, 0, N,
                backend=backend, force_dim_axis=force_dim_axis)


class ChebyshevGaussQuadrature(JacobiGaussQuadrature):
    r"""A Gauss quadrature rule with weight :math:`\sqrt{1-x^2}^{\mp 1}`
    and *N+1* nodes.

    The Chebyshev-Gauss quadrature rules of the first kind and second kind
    correspond to Gauss-Jacobi quadrature rules with
    :math:`\alpha = \beta = -0.5` and :math:`\alpha = \beta = 0.5`,
    respectively.

    .. versionadded:: 2019.1
    """

    def __init__(self,
                 N: int,  # noqa: N803
                 kind: int = 1,
                 backend: str | None = None,
                 force_dim_axis: bool = False) -> None:
        if kind == 1:
            alpha = beta = -0.5
        elif kind == 2:
            alpha = beta = +0.5
        else:
            raise ValueError(f"unsupported kind: '{kind}'")

        super().__init__(
                alpha, beta, N,
                backend=backend, force_dim_axis=force_dim_axis)


class GaussGegenbauerQuadrature(JacobiGaussQuadrature):
    r"""Gauss-Gegenbauer quadrature is a special case of Gauss-Jacobi quadrature
    with :math:`\alpha = \beta` and *N+1* nodes.

    .. versionadded:: 2019.1
    """

    def __init__(self,
                 alpha: float, N: int,  # noqa: N803
                 backend: str | None = None,
                 force_dim_axis: bool = False) -> None:
        super().__init__(
                alpha, alpha, N,
                backend=backend, force_dim_axis=force_dim_axis)


def jacobi_gauss_lobatto_nodes(
        alpha: float, beta: float, N: int,          # noqa: N803
        backend: str | None = None,
        force_dim_axis: bool = False) -> NDArray[np.floating]:
    """Compute the Gauss-Lobatto quadrature
    nodes corresponding to the :class:`~modepy.JacobiGaussQuadrature`
    with the same parameters. There will be *N+1* nodes.

    Exact to degree :math:`2N - 3`.
    """

    x: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = np.zeros((N + 1,))
    if N == 0:
        x[0] = 0
        return x

    x[0] = -1
    x[-1] = 1

    if N > 1:
        quad = JacobiGaussQuadrature(alpha + 1, beta + 1, N - 2,
                backend=backend, force_dim_axis=True)
        x[1:-1] = quad.nodes[0].real

    if force_dim_axis:
        x = x.reshape(1, -1)

    return x


def legendre_gauss_lobatto_nodes(
        N: int,                     # noqa: N803
        backend: str | None = None,
        force_dim_axis: bool = False) -> NDArray[np.floating]:
    """Compute the Legendre-Gauss-Lobatto quadrature nodes.
    *N+1* is the number of nodes.

    Exact to degree :math:`2N - 1`.
    """
    return jacobi_gauss_lobatto_nodes(0, 0, N,
            backend=backend, force_dim_axis=force_dim_axis)


class JacobiGaussLobattoQuadrature(Quadrature):
    r"""
    Compute the Jacobi-Gauss-Lobatto quadrature with respect
    to the Jacobi weight function

    .. math::

        I[f] = \int_{-1}^1 f(x) (1 - x)^\alpha (1 + x)^\beta\, \mathrm{d}x,

    There will be *N+1* nodes. Exact to degree :math:`2N - 3`.

    .. versionadded:: 2024.2
    """

    def __init__(self,
            alpha: float, beta: float, N: int,  # noqa: N803
            *, backend: str | None = None,
            force_dim_axis: bool = False) -> None:
        nodes = jacobi_gauss_lobatto_nodes(alpha, beta, N, backend)

        from math import gamma

        from modepy.modes import binom, scaled_jacobi

        # Formula numbers refer to https://doi.org/10.1023/A:1016689830453
        if N + 1 < 2:
            raise ValueError("Lobatto rules must have at least two nodes")

        # Alternate reference via chebfun, using same source paper but
        # using the beta function in implementation:
        # https://github.com/chebfun/chebfun/blob/db207bc9f48278ca4def15bf90591bfa44d0801d/lobpts.m

        n = N - 1

        # FIXME: Recurrences might be better than just typing up the formulas.

        # (3.10)
        common_factor = (
            2**(alpha + beta + 1)
            * binom(n + alpha + 1, n)
            / binom(n + beta + 1, n)
            / binom(n + alpha + beta + 2, n)
            * gamma(alpha + 2)
            / gamma(alpha + beta + 3)
        )
        edge_weight = (
            common_factor
            * gamma(beta + 1)
        )

        # (4.7)
        int_nodes = nodes[1:-1]
        frac = (
            4*(n+alpha+1) * (n+beta+1) + (alpha-beta)**2
        ) / (
            2*n + alpha + beta + 2
        )**2

        int_weights = (
            common_factor
            * gamma(beta + 2)

            # FIXME: This is part of the paper, but tests only pass without it:
            # / (n + 1)**2

            / (frac - int_nodes**2)
            * (1 - int_nodes**2)
            / scaled_jacobi(alpha, beta, n + 1, int_nodes)**2
        )

        weights = np.empty_like(nodes)
        weights[0] = edge_weight
        weights[-1] = edge_weight
        weights[1:-1] = int_weights

        if force_dim_axis:
            nodes = nodes.reshape(1, -1)

        super().__init__(nodes, weights, 2*N - 3)


class LegendreGaussLobattoQuadrature(JacobiGaussLobattoQuadrature):
    def __init__(
                self, N, *, backend: str | None = None,  # noqa: N803
                force_dim_axis: bool = False
            ) -> None:
        super().__init__(0, 0, N, backend=backend, force_dim_axis=force_dim_axis)
