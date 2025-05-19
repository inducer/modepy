from __future__ import annotations

from numpy.typing import NDArray


__copyright__ = "Copyright (C) 2009-2013 Andreas Kloeckner"

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

from functools import reduce

import numpy as np

from modepy.quadrature import Quadrature, inf


def _extended_euclidean(q: int, r: int) -> tuple[int, int, int]:
    """Return a tuple (p, a, b) such that p = aq + br,
    where p is the greatest common divisor.
    """

    # see [Davenport], Appendix, p. 214

    if abs(q) < abs(r):
        p, a, b = _extended_euclidean(r, q)
        return p, b, a

    Q = (1, 0)  # noqa: N806
    R = (0, 1)  # noqa: N806

    while r:
        quot, t = divmod(q, r)
        T = Q[0] - quot*R[0], Q[1] - quot*R[1]  # noqa: N806
        q, r = r, t
        Q, R = R, T     # noqa: N806

    return q, Q[0], Q[1]


def _gcd(q: int, r: int) -> int:
    return _extended_euclidean(q, r)[0]


def _simplify_fraction(xxx_todo_changeme: tuple[int, int]) -> tuple[int, int]:
    (a, b) = xxx_todo_changeme
    gcd = _gcd(a, b)
    return (a//gcd, b//gcd)


class GrundmannMoellerSimplexQuadrature(Quadrature):
    r"""Cubature on an *n*-simplex from [Grundmann1978]_.

    This cubature rule has both negative and positive weights. It is exact for
    polynomials up to order :math:`2s + 1`, where :math:`s` is given as *order*.
    The integration domain is the unit simplex (see :ref:`tri-coords`
    and :ref:`tet-coords`).

    .. [Grundmann1978] A. Grundmann and H.M. Moeller,
        *Invariant integration formulas for the n-simplex by combinatorial methods*,
        SIAM J. Numer. Anal. 15 (1978), 282--290.
        `DOI <http://dx.doi.org/10.1137/0715019>`__

    .. automethod:: __init__
    .. automethod:: __call__
    """

    # FIXME: most other functionality in modepy uses 'dims, order' as the
    # argument order convention.
    def __init__(self, order: int, dimension: int) -> None:
        """
        :arg order: a parameter correlated with the total degree of polynomials
            that are integrated exactly (see also
            :attr:`~modepy.Quadrature.exact_to`).
        :arg dimension: the number of dimensions for the quadrature rule.
        """
        s = order
        n = dimension
        d = 2*s + 1

        if dimension == 0:
            super().__init__(np.zeros((dimension, 1)), np.ones(1), exact_to=inf)
            return

        import math

        from pytools import (
            generate_decreasing_nonnegative_tuples_summing_to,
            generate_unique_permutations,
            wandering_element,
        )

        points_to_weights: dict[tuple[tuple[int, int], ...], NDArray[np.floating]] = {}

        for i in range(s + 1):
            weight = (-1)**i * 2**(-2*s) \
                    * (d + n - 2*i)**d \
                    / math.factorial(i) \
                    / math.factorial(d + n - i)

            for t in generate_decreasing_nonnegative_tuples_summing_to(s - i, n + 1):
                for beta in generate_unique_permutations(t):
                    denominator = d + n - 2*i
                    point = tuple(
                            _simplify_fraction((2*beta_i + 1, denominator))
                            for beta_i in beta)

                    points_to_weights[point] = \
                            points_to_weights.get(point, 0) + weight

        from operator import add

        vertices = ([-1 * np.ones((n,))]
                + [np.array(x)
                    for x in wandering_element(n, landscape=-1, wanderer=1)])

        nodes: list[NDArray[np.floating]] = []
        weights: list[NDArray[np.floating]] = []

        dim_factor = 2**n
        for p, w in points_to_weights.items():
            real_p = reduce(add, (a/b * v
                                for (a, b), v in zip(p, vertices, strict=True)))
            nodes.append(real_p)
            weights.append(dim_factor * w)

        super().__init__(np.array(nodes).T, np.array(weights), exact_to=d)
