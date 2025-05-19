"""
.. currentmodule:: modepy

.. autoclass:: Quadrature
    :members:
    :special-members: __call__

.. autoclass:: ZeroDimensionalQuadrature
    :show-inheritance:

.. autofunction:: quadrature_for_space

.. currentmodule:: modepy.quadrature

.. class:: Quadrature

    See :class:`modepy.Quadrature`.

.. class:: _Inf

    A sentinel type for infinite results. Do not use directly. Use
    :func:`isinf` instead.

.. autofunction:: isinf
"""

from __future__ import annotations



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

from collections.abc import Callable, Iterable, Sequence
from functools import singledispatch
from numbers import Number

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Never

from modepy.shapes import Shape, Simplex, TensorProductShape
from modepy.spaces import PN, FunctionSpace, TensorProductSpace


class QuadratureRuleUnavailable(RuntimeError):
    """

    .. versionadded :: 2013.3
    """


# Literal(float("inf")) might have been nicer, but alas:
# https://github.com/python/typing/issues/1160
class _Inf:
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Inf)

    def __gt__(self, other: object) -> bool:
        if isinstance(other, _Inf):
            return False
        return bool(isinstance(other, Number))

    def __ge__(self, other: object) -> bool:
        return bool(isinstance(other, Number | _Inf))

    def __add__(self, other: object) -> _Inf:
        if (isinstance(other, float | int | np.integer | np.floating)
                and np.isfinite(other)):
            return self

        return NotImplemented

    def __index__(self) -> Never:
        raise ValueError("_Inf cannot be cast to integer")


inf = _Inf()


def isinf(obj: object) -> bool:
    return isinstance(obj, _Inf)


class Quadrature:
    """The basic interface for a quadrature rule."""

    def __init__(self,
                 nodes: NDArray[np.inexact],
                 weights: NDArray[np.inexact],
                 exact_to: int | _Inf | None = None) -> None:
        """
        :arg nodes: an array of shape *(d, nnodes)*, where *d* is the dimension
            of the qudrature rule.
        :arg weights: an array of length *nnodes*.
        :arg exact_to: an optional argument denoting the summed polynomial
            degree to which the quadrature is exact. By default, `exact_to`
            is `None` and will *not* be set as an attribute.
        """

        self.nodes: NDArray[np.inexact] = nodes
        """An array of shape *(dim, nnodes)*, where *dim* is the dimension
        of the qudrature rule. In 1D, the shape is just *(nnodes,)*.
        """
        self.weights: NDArray[np.inexact] = weights
        """A vector of length *nnodes* that contains the quadrature weights."""
        self._exact_to = exact_to

    @property
    def dim(self) -> int:
        """Dimension of the space on which the quadrature rule applies."""
        return 1 if self.nodes.ndim == 1 else self.nodes.shape[0]

    @property
    def exact_to(self) -> int | _Inf:
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

    def __call__(
                self,
                f: Callable[[NDArray[np.inexact]], NDArray[np.inexact]]
            ) -> NDArray[np.inexact] | np.inexact:
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

    .. autoattribute:: quadratures
    .. automethod:: __init__
    """

    quadratures: Sequence[Quadrature]
    """The lower-dimensional quadratures from which the tensor product quadrature is
    composed.
    """

    def __init__(self, quads: Iterable[Quadrature]) -> None:
        """
        :arg quad: a iterable of :class:`Quadrature` objects for one-dimensional
            intervals, one for each dimension of the tensor product.
        """

        from modepy.nodes import tensor_product_nodes

        quads = tuple(quads)
        x = tensor_product_nodes([quad.nodes for quad in quads])
        w = np.prod(tensor_product_nodes([quad.weights for quad in quads]), axis=0)
        assert w.size == x.shape[1]

        if quads:
            try:
                exact_to = min(quad.exact_to for quad in quads)
            except AttributeError:
                # e.g. FejerQuadrature does not have any 'exact_to'
                exact_to = None
        else:
            # 0D quadrature is point evaluation
            # "infinite" accuracy
            exact_to = inf

        super().__init__(x, w, exact_to=exact_to)

        self.quadratures = quads


class LegendreGaussTensorProductQuadrature(TensorProductQuadrature):
    """A tensor product using only :class:`~modepy.LegendreGaussQuadrature`
    one-dimenisonal rules.
    """

    def __init__(self,
                 N: int, dims: int,  # noqa: N803
                 backend: str | None = None) -> None:
        from modepy.quadrature.jacobi_gauss import LegendreGaussQuadrature
        super().__init__([
            LegendreGaussQuadrature(N, backend=backend, force_dim_axis=True)
            ] * dims)


class LegendreGaussLobattoTensorProductQuadrature(TensorProductQuadrature):
    """A tensor product using only :class:`~modepy.LegendreGaussLobattoQuadrature`
    one-dimenisonal rules.
    """

    def __init__(self,
                 N: int, dims: int,  # noqa: N803
                 backend: str | None = None) -> None:
        from modepy.quadrature.jacobi_gauss import LegendreGaussLobattoQuadrature
        super().__init__([
            LegendreGaussLobattoQuadrature(N, backend=backend, force_dim_axis=True)
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
            for sp, shp in zip(space.bases, shape.bases, strict=True)
            ])

    assert all(quad.exact_to >= getattr(s, "order", 0) for s in space.bases)
    return quad

# }}}

# vim: foldmethod=marker
