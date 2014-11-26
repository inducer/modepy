from __future__ import division
from __future__ import absolute_import
from six.moves import range

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
from modepy.quadrature import Quadrature




class JacobiGaussQuadrature(Quadrature):
    r"""An Gauss quadrature of order *N* associated with the
    Jacobi weight :math:`(1-x)^\alpha(1+x)^\beta`.

    Assumes :math:`\alpha,\beta > -1` with
    :math:`(\alpha,\beta)\not\in\{(-1/2,-1/2)\}`.

    Integrates on the interval (-1,1).
    The quadrature rule is exact up to degree :math:`2N+1`.

    Inherits from :class:`modepy.Quadrature`. See there for the interface
    to obtain nodes and weights.
    """
    def __init__(self, alpha, beta, N):
        x, w = self.compute_weights_and_nodes(N, alpha, beta)
        Quadrature.__init__(self, x, w)

    @staticmethod
    def compute_weights_and_nodes(N, alpha, beta):
        """Return (nodes, weights) for an n-th order Gauss quadrature
        with the Jacobi polynomials of type (alpha, beta).
        """

        # FIXME: This could stand to be upgraded to the Rokhlin algorithm.

        # follows
        # Gene H. Golub, John H. Welsch, Calculation of Gauss Quadrature Rules,
        # Mathematics of Computation, Vol. 23, No. 106 (Apr., 1969), pp. 221-230
        # http://dx.doi.org/10.2307/2004418

        # see also doc/hedge-notes.tm for correspondence with the Jacobi
        # recursion from Hesthaven/Warburton's book

        from math import sqrt

        apb = alpha+beta

        # see Appendix A of Hesthaven/Warburton for these formulas
        def a(n):
            return (
                    2/(2*n+apb)
                    *
                    sqrt(
                        (n*(n+apb)*(n+alpha)*(n+beta))
                        /
                        ((2*n+apb-1)*(2*n+apb+1))
                        )
                    )

        def b(n):
            if n == 0:
                return (
                        -(alpha-beta)
                        /
                        (apb+2)
                        )
            else:
                return (
                        -(alpha**2-beta**2)
                        /
                        ((2*n+apb)*(2*n+apb+2))
                        )

        T = np.zeros((N+1, N+1))

        for n in range(N+1):
            T[n,n] = b(n)
            if n > 0:
                T[n,n-1] = current_a
            if n < N:
                next_a = a(n+1)
                T[n,n+1] = next_a
                current_a = next_a

        assert la.norm(T-T.T) < 1e-12
        eigval, eigvec = la.eigh(T)

        assert la.norm(np.dot(T, eigvec) - np.dot(eigvec, np.diag(eigval))) < 1e-12

        from modepy.modes import jacobi
        from functools import partial
        p0 = partial(jacobi, alpha, beta, 0) # that's a constant, sure
        nodes = eigval
        weights = np.array([eigvec[0,i]**2 / p0(nodes[i])**2 for i in range(N+1)])

        return nodes, weights




class LegendreGaussQuadrature(JacobiGaussQuadrature):
    """An Gauss quadrature associated with weight 1.

    Integrates on the interval (-1,1).
    The quadrature rule is exact up to degree :math:`2N+1`.

    Inherits from :class:`modepy.Quadrature`. See there for the interface
    to obtain nodes and weights.
    """
    def __init__(self, N):
        JacobiGaussQuadrature.__init__(self, 0, 0, N)




def jacobi_gauss_lobatto_nodes(alpha, beta, N):
    """Compute the Gauss-Lobatto quadrature
    nodes corresponding to the :class:`JacobiGaussQuadrature`
    with the same parameters.

    Exact to degree :math:`2N-3`.
    """

    x = np.zeros((N+1,))
    x[0] = -1
    x[-1] = 1

    if N == 1:
        return x

    x[1:-1] = np.array(
            JacobiGaussQuadrature(alpha+1, beta+1, N-2).nodes
            ).real
    return x




def legendre_gauss_lobatto_nodes(N):
    """Compute the Legendre-Gauss-Lobatto quadrature nodes.

    Exact to degree :math:`2N-1`.
    """
    return jacobi_gauss_lobatto_nodes(0, 0, N)
