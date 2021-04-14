"""
.. currentmodule:: modepy

.. autoclass:: Quadrature
.. autoclass:: ZeroDimensionalQuadrature

.. autofunction:: quadrature_for_space

Redirections to Canonical Names
-------------------------------

.. currentmodule:: modepy.quadrature

.. class:: Quadrature

    See :class:`modepy.Quadrature`.
"""

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

from functools import singledispatch

import numpy as np
from modepy.shapes import Shape, Simplex, Hypercube
from modepy.spaces import FunctionSpace, PN, QN


class QuadratureRuleUnavailable(RuntimeError):
    """

    .. versionadded :: 2013.3
    """


class Quadrature:
    """The basic interface for a quadrature rule.

    .. attribute :: nodes

        An array of shape *(d, nnodes)*, where *d* is the dimension
        of the qudrature rule. In 1D, the shape is just *(nnodes,)*.

    .. attribute :: weights

        A vector of length *nnodes*.

    .. attribute :: exact_to

        Summed polynomial degree up to which the quadrature is exact.
        In higher-dimensions, the quadrature is supposed to be exact on (at least)
        :math:`P^N`, where :math:`N` = :attr:`exact_to`. If the quadrature
        accuracy is not known, attr:`exact_to` will *not* be set, and
        an `AttributeError` will be raised when attempting to access this
        information.

    .. automethod:: __init__

    .. automethod:: __call__
    """

    def __init__(self, nodes, weights, exact_to = None):
        """
        :arg nodes: an array of shape *(d, nnodes)*, where *d* is the dimension
            of the qudrature rule.
        :arg weights: an array of length *nnodes*.
        :arg exact_to: an optional argument denoting the symmed polynomial
            degree to which the quadrature is exact. By default, `exact_to`
            is `None` and will *not* be set as an attribute.
        """
        self.nodes = nodes
        self.weights = weights
        # TODO: May be revamped/addressed later;
        # see https://github.com/inducer/modepy/issues/31
        if exact_to:
            self.exact_to = exact_to

    def __call__(self, f):
        """Evaluate the callable *f* at the quadrature nodes and return its
        integral.

        *f* is assumed to accept arrays of shape *(dims, npts)*,
        or of shape *(npts,)* for 1D quadrature.
        """
        return np.dot(self.weights, f(self.nodes))


class ZeroDimensionalQuadrature(Quadrature):
    """A quadrature rule that should be used for 0d domains (i.e. points).

    Inherits from :class:`Quadrature`.
    """

    def __init__(self):
        self.nodes = np.empty((0, 1), dtype=np.float64)
        self.weights = np.ones((1,), dtype=np.float64)
        self.exact_to = np.inf


class Transformed1DQuadrature(Quadrature):
    """A quadrature rule on an arbitrary interval :math:`(a, b)`."""

    def __init__(self, quad, left, right):
        """Transform a given 1D quadrature rule *quad* onto the
        interval (left, right).
        """
        self.left = left
        self.right = right

        length = right - left
        half_length = length / 2
        assert length > 0

        Quadrature.__init__(self,
                left + (quad.nodes+1) / 2 * length,
                quad.weights * half_length)


class TensorProductQuadrature(Quadrature):
    """
    .. automethod:: __init__
    """

    def __init__(self, quads):
        """
        :arg quad: a :class:`tuple` of :class:`Quadrature` for one-dimensional
            intervals, one for each dimension of the tensor product.
        """

        from modepy.nodes import tensor_product_nodes
        x = tensor_product_nodes([quad.nodes for quad in quads])
        w = np.prod(tensor_product_nodes([quad.weights for quad in quads]), axis=0)
        assert w.size == x.shape[1]

        super().__init__(x, w)

        try:
            exact_to = min(quad.exact_to for quad in quads)
        except AttributeError:
            # e.g. FejerQuadrature does not have any 'exact_to'
            pass
        else:
            self.exact_to = exact_to


class LegendreGaussTensorProductQuadrature(TensorProductQuadrature):
    def __init__(self, N, dims, backend=None):      # noqa: N803
        from modepy.quadrature.jacobi_gauss import LegendreGaussQuadrature
        super().__init__([LegendreGaussQuadrature(N, backend=backend)] * dims)


# {{{ quadrature

@singledispatch
def quadrature_for_space(space: FunctionSpace, shape: Shape) -> Quadrature:
    """
    :returns: a :class:`~modepy.Quadrature` that exactly integrates the functions
        in *space* over *shape*.
    """
    raise NotImplementedError((type(space).__name__, type(shape).__name))


@quadrature_for_space.register(PN)
def _(space: PN, shape: Simplex):
    if not isinstance(shape, Simplex):
        raise NotImplementedError((type(space).__name__, type(shape).__name))
    if space.spatial_dim != shape.dim:
        raise ValueError("spatial dimensions of shape and space must match")

    import modepy as mp
    if space.spatial_dim == 0:
        quad = ZeroDimensionalQuadrature()
    else:
        try:
            quad = mp.XiaoGimbutasSimplexQuadrature(space.order, space.spatial_dim)
        except QuadratureRuleUnavailable:
            quad = mp.GrundmannMoellerSimplexQuadrature(
                    space.order//2, space.spatial_dim)

    assert quad.exact_to >= space.order
    return quad


@quadrature_for_space.register(QN)
def _(space: QN, shape: Hypercube):
    if not isinstance(shape, Hypercube):
        raise NotImplementedError((type(space).__name__, type(shape).__name))
    if space.spatial_dim != shape.dim:
        raise ValueError("spatial dimensions of shape and space must match")

    if space.spatial_dim == 0:
        quad = ZeroDimensionalQuadrature()
    else:
        quad = LegendreGaussTensorProductQuadrature(space.order, space.spatial_dim)

    assert quad.exact_to >= space.order
    return quad

# }}}

# vim: foldmethod=marker
