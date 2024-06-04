"""
.. currentmodule:: modepy

.. autoclass:: Quadrature
    :members:
    :special-members: __call__

.. autoclass:: ZeroDimensionalQuadrature
    :show-inheritance:

.. autofunction:: quadrature_for_space
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
from typing import Callable, Iterable, Optional

import numpy as np

from modepy.shapes import Shape, Simplex, TensorProductShape
from modepy.spaces import PN, FunctionSpace, TensorProductSpace


class QuadratureRuleUnavailable(RuntimeError):
    """

    .. versionadded :: 2013.3
    """


class Quadrature:
    """The basic interface for a quadrature rule."""

    def __init__(self,
                 nodes: np.ndarray,
                 weights: np.ndarray,
                 exact_to: Optional[int] = None) -> None:
        """
        :arg nodes: an array of shape *(d, nnodes)*, where *d* is the dimension
            of the qudrature rule.
        :arg weights: an array of length *nnodes*.
        :arg exact_to: an optional argument denoting the summed polynomial
            degree to which the quadrature is exact. By default, `exact_to`
            is `None` and will *not* be set as an attribute.
        """

        self.nodes: np.ndarray = nodes
        """An array of shape *(dim, nnodes)*, where *dim* is the dimension
        of the qudrature rule. In 1D, the shape is just *(nnodes,)*.
        """
        self.weights: np.ndarray = weights
        """A vector of length *nnodes* that contains the quadrature weights."""
        self._exact_to = exact_to

    @property
    def dim(self) -> int:
        """Dimension of the space on which the quadrature rule applies."""
        return 1 if self.nodes.ndim == 1 else self.nodes.shape[0]

    @property
    def exact_to(self) -> int:
        """Summed polynomial degree up to which the quadrature is exact.

        In higher-dimensions, the quadrature is supposed to be exact on (at least)
        :math:`P^N`, where :math:`N` = :attr:`exact_to`. If the quadrature
        accuracy is not known, :attr:`exact_to` will *not* be set, and
        an :exc:`AttributeError` will be raised when attempting to access this
        information.
        """

        # TODO: May be revamped/addressed later:
        # see https://github.com/inducer/modepy/issues/31
        if self._exact_to is not None:
            return self._exact_to

        raise AttributeError(
            f"'{type(self).__name__}' rule does not define an exact order"
            )

    def __call__(self, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Evaluate the callable *f* at the quadrature nodes and return its
        integral.

        *f* is assumed to accept arrays of shape *(dims, npts)*,
        or of shape *(npts,)* for 1D quadrature.
        """
        return np.dot(self.weights, f(self.nodes).T).T


class ZeroDimensionalQuadrature(Quadrature):
    """A quadrature rule that should be used for 0d domains (i.e. points)."""

    def __init__(self) -> None:
        super().__init__(np.empty((0, 1), dtype=np.float64),
                         np.ones((1,), dtype=np.float64),
                         # NOTE: exact_to is expected to be an int
                         exact_to=np.inf)   # type: ignore[arg-type]


class Transformed1DQuadrature(Quadrature):
    """A quadrature rule on an arbitrary interval :math:`[a, b]`."""

    def __init__(self, quad: Quadrature, left: float, right: float) -> None:
        self.left = left
        """Left bound of the transformed interval."""
        self.right = right
        """Right bound of the transformed interval."""

        length = right - left
        if length <= 0:
            raise ValueError(
                f"Transformed interval cannot be empty: [{left}, {right}]"
                )

        super().__init__(
            left + (quad.nodes + 1) / 2 * length,
            length / 2 * quad.weights,
            exact_to=quad._exact_to)


class TensorProductQuadrature(Quadrature):
    r"""A tensor product quadrature of one-dimensional :class:`Quadrature`\ s.

    .. automethod:: __init__
    """

    def __init__(self, quads: Iterable[Quadrature]) -> None:
        """
        :arg quad: a iterable of :class:`Quadrature` objects for one-dimensional
            intervals, one for each dimension of the tensor product.
        """

        from modepy.nodes import tensor_product_nodes

        quads = list(quads)
        x = tensor_product_nodes([quad.nodes for quad in quads])
        w = np.prod(tensor_product_nodes([quad.weights for quad in quads]), axis=0)
        assert w.size == x.shape[1]

        try:
            exact_to = min(quad.exact_to for quad in quads)
        except AttributeError:
            # e.g. FejerQuadrature does not have any 'exact_to'
            exact_to = None

        super().__init__(x, w, exact_to=exact_to)


class LegendreGaussTensorProductQuadrature(TensorProductQuadrature):
    """A tensor product using only :class:`~modepy.LegendreGaussQuadrature`
    one-dimenisonal rules.
    """

    def __init__(self,
                 N: int, dims: int,  # noqa: N803
                 backend: Optional[str] = None) -> None:
        from modepy.quadrature.jacobi_gauss import LegendreGaussQuadrature
        super().__init__([
            LegendreGaussQuadrature(N, backend=backend, force_dim_axis=True)
            ] * dims)


# {{{ quadrature

@singledispatch
def quadrature_for_space(space: FunctionSpace, shape: Shape) -> Quadrature:
    """
    :returns: a :class:`~modepy.Quadrature` that exactly integrates the functions
        in *space* over *shape*.
    """
    raise NotImplementedError((type(space).__name__, type(shape).__name__))


@quadrature_for_space.register(PN)
def _quadrature_for_pn(space: PN, shape: Simplex) -> Quadrature:
    if not isinstance(shape, Simplex):
        raise NotImplementedError((type(space).__name__, type(shape).__name__))
    if space.spatial_dim != shape.dim:
        raise ValueError("spatial dimensions of shape and space must match")

    import modepy as mp
    if space.spatial_dim == 0:
        quad: Quadrature = ZeroDimensionalQuadrature()
    elif space.spatial_dim == 1:
        quad = mp.LegendreGaussQuadrature(space.order, force_dim_axis=True)
    else:
        try:
            quad = mp.XiaoGimbutasSimplexQuadrature(space.order, space.spatial_dim)
        except QuadratureRuleUnavailable:
            quad = mp.GrundmannMoellerSimplexQuadrature(
                    space.order//2, space.spatial_dim)

    assert quad.exact_to >= space.order
    return quad


@quadrature_for_space.register(TensorProductSpace)
def _quadrature_for_tp(
            space: TensorProductSpace,
            shape: TensorProductShape
        ) -> Quadrature:
    if not isinstance(shape, TensorProductShape):
        raise NotImplementedError((type(space).__name__, type(shape).__name))

    if space.spatial_dim != shape.dim:
        raise ValueError("spatial dimensions of shape and space must match")

    if space.spatial_dim == 0:
        quad: Quadrature = ZeroDimensionalQuadrature()
    else:
        quad = TensorProductQuadrature([
            quadrature_for_space(sp, shp)
            for sp, shp in zip(space.bases, shape.bases)
            ])

    assert all(quad.exact_to >= getattr(s, "order", 0) for s in space.bases)
    return quad

# }}}

# vim: foldmethod=marker
