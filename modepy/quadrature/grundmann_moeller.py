from __future__ import division, absolute_import

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

import six
from six.moves import range
from six.moves import zip
from functools import reduce

import numpy as np
from modepy.quadrature import Quadrature


def _extended_euclidean(q, r):
    """Return a tuple (p, a, b) such that p = aq + br,
    where p is the greatest common divisor.
    """

    # see [Davenport], Appendix, p. 214

    if abs(q) < abs(r):
        p, a, b = _extended_euclidean(r, q)
        return p, b, a

    Q = 1, 0
    R = 0, 1

    while r:
        quot, t = divmod(q, r)
        T = Q[0] - quot*R[0], Q[1] - quot*R[1]
        q, r = r, t
        Q, R = R, T

    return q, Q[0], Q[1]


def _gcd(q, r):
    return _extended_euclidean(q, r)[0]


def _simplify_fraction(xxx_todo_changeme):
    (a, b) = xxx_todo_changeme
    gcd = _gcd(a, b)
    return (a//gcd, b//gcd)


class GrundmannMoellerSimplexQuadrature(Quadrature):
    r"""Cubature on an *n*-simplex.

    This cubature rule has both negative and positive weights.
    It is exact for polynomials up to order :math:`2s+1`, where
    :math:`s` is given as *order*.

    The integration domain is the unit simplex. (see :ref:`tri-coords`
    and :ref:`tet-coords`)

    .. attribute:: exact_to

        The total degree up to which the quadrature is exact.

    See

    * A. Grundmann and H.M. Moeller,
      Invariant integration formulas for the n-simplex by combinatorial methods,
      SIAM J. Numer. Anal.  15 (1978), 282--290.
      http://dx.doi.org/10.1137/0715019

    """

    def __init__(self, order, dimension):
        """
        :arg order: A parameter correlated with the total degree of polynomials
            that are integrated exactly. (See also :attr:`exact_to`.)
        :arg dimension: The number of dimensions for the quadrature rule.
            Any positive integer.
        """
        s = order
        n = dimension
        d = 2*s+1

        self.exact_to = d

        if dimension == 0:
            nodes = np.zeros((dimension, 1))
            weights = np.ones(1)

            Quadrature.__init__(self, nodes, weights)
            return

        from pytools import \
                generate_decreasing_nonnegative_tuples_summing_to, \
                generate_unique_permutations, \
                factorial, \
                wandering_element

        points_to_weights = {}

        for i in range(s+1):
            weight = (-1)**i * 2**(-2*s) \
                    * (d + n-2*i)**d \
                    / factorial(i) \
                    / factorial(d+n-i)

            for t in generate_decreasing_nonnegative_tuples_summing_to(s-i, n+1):
                for beta in generate_unique_permutations(t):
                    denominator = d+n-2*i
                    point = tuple(
                            _simplify_fraction((2*beta_i+1, denominator))
                            for beta_i in beta)

                    points_to_weights[point] = \
                            points_to_weights.get(point, 0) + weight

        from operator import add

        vertices = [-1 * np.ones((n,))] \
                + [np.array(x)
                        for x in wandering_element(n, landscape=-1, wanderer=1)]

        nodes = []
        weights = []

        dim_factor = 2**n
        for p, w in six.iteritems(points_to_weights):
            real_p = reduce(add, (a/b*v for (a, b), v in zip(p, vertices)))
            nodes.append(real_p)
            weights.append(dim_factor*w)

        Quadrature.__init__(self, np.array(nodes).T, np.array(weights))
