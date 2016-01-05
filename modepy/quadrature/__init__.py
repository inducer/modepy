from __future__ import division

__copyright__ = ("Copyright (C) 2009, 2010, 2013 Andreas Kloeckner, Tim Warburton, "
        "Jan Hesthaven, Xueyu Zhu")

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


class QuadratureRuleUnavailable(RuntimeError):
    """

    .. versionadded :: 2013.3
    """


class Quadrature(object):
    """The basic interface for a quadrature rule.

    .. attribute :: nodes

        An array of shape *(d, nnodes)*, where *d* is the dimension
        of the qudrature rule. In 1D, the shape is just *(nnodes,)*.

    .. attribute :: weights

        A vector of length *nnodes*.
    """
    def __init__(self, nodes, weights):
        self.nodes = nodes
        self.weights = weights

    def __call__(self, f):
        """Evaluate the callable *f* at the quadrature nodes and return its
        integral.

        *f* is assumed to accept arrays of shape *(dims, npts)*,
        or of shape *(npts,)* for 1D quadrature.
        """
        return np.dot(self.weights, f(self.nodes))


class Transformed1DQuadrature(Quadrature):
    """A quadrature rule on an arbitrary interval :math:`(a,b)`. """

    def __init__(self, quad, left, right):
        """Transform a given 1D quadrature rule *quad* onto the
        interval (left, right).
        """
        self.left = left
        self.right = right

        length = right-left
        assert length > 0
        half_length = length / 2
        Quadrature.__init__(self,
                left + (quad.nodes+1)/2*length,
                quad.weights*half_length)
