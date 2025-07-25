"""
Function Spaces
---------------

.. autoclass:: FunctionSpaceT

.. currentmodule:: modepy

.. autoclass:: FunctionSpace
    :members:

.. autoclass:: TensorProductSpace
    :members:
    :show-inheritance:

.. autoclass:: PN
    :members:
    :show-inheritance:

.. autoclass:: QN
    :members:
    :show-inheritance:

.. autofunction:: space_for_shape
"""
from __future__ import annotations


__copyright__ = "Copyright (C) 2020 Andreas Kloeckner"

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

from abc import ABC, abstractmethod
from functools import singledispatch
from numbers import Number
from typing import Literal, TypeVar, overload

import numpy as np
from typing_extensions import override

from modepy.shapes import Shape, Simplex, TensorProductShape


# {{{ function spaces

FunctionSpaceT = TypeVar("FunctionSpaceT", bound="FunctionSpace")
"""An invariant generic variable bound to :class:`FunctionSpace`."""


class FunctionSpace(ABC):
    r"""An opaque object representing a finite-dimensional function space
    of functions :math:`\mathbb{R}^n \to \mathbb{R}`.
    """

    @property
    def order(self) -> int:
        """Polynomial degree of the function space, if any."""
        raise AttributeError(f"{type(self).__name__} has no attribute 'order'")

    @property
    @abstractmethod
    def spatial_dim(self) -> int:
        """The number of spatial dimensions in which the functions in the space
        operate (:math:`n` in the above definition).
        """

    @property
    @abstractmethod
    def space_dim(self) -> int:
        """The number of dimensions of the function space."""


@singledispatch
def space_for_shape(
        shape: Shape,
        order: int | tuple[int, ...]
        ) -> FunctionSpace:
    r"""Return an unspecified instance of :class:`FunctionSpace` suitable
    for approximation on *shape* attaining interpolation error of
    :math:`O(h^{\text{order}+1})`.

    :arg order: an integer interpolation order or a :class:`tuple` of orders.
        Taking a :class:`tuple` of orders is not supported by all function
        spaces. A notable exception being :class:`TensorProductSpace`,
        which allows defining different orders for each base space.
    """

    raise NotImplementedError(type(shape).__name__)

# }}}


# # {{{ generic tensor product space

class TensorProductSpace(FunctionSpace):
    """A function space defined as the tensor product of lower dimensional spaces.

    To recover the tensor product structure of degree-of-freedom arrays (nodal
    or modal) associated with this type of space, see
    :func:`~modepy.tools.reshape_array_for_tensor_product_space`.
    """

    bases: tuple[FunctionSpace, ...]
    """A :class:`tuple` of the base spaces that take part in the tensor product."""

    @overload
    # pyright-ignore: they overlap, can't be helped.
    def __new__(cls, bases: tuple[FunctionSpaceT]) -> FunctionSpaceT: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __new__(cls, bases: tuple[FunctionSpace, ...]) -> TensorProductSpace: ...

    def __new__(cls, bases: tuple[FunctionSpace, ...]) -> FunctionSpace:
        if len(bases) == 1:
            return bases[0]
        else:
            return FunctionSpace.__new__(cls)

    def __init__(self, bases: tuple[FunctionSpace, ...]) -> None:
        self.bases = sum((
            space.bases if isinstance(space, TensorProductSpace) else (space,)
            for space in bases
            ), ())

    def __getnewargs__(self):
        # Ensures TensorProductSpace is picklable
        return (self.bases,)

    @property
    @override
    def order(self) -> int:
        """Polynomial degree of the functions in the space, if any."""

        from pytools import is_single_valued
        if is_single_valued([space.order for space in self.bases]):
            return self.bases[0].order
        else:
            raise AttributeError(f"{type(self).__name__} has no attribute 'order'")

    @property
    @override
    def spatial_dim(self) -> int:
        return sum(space.spatial_dim for space in self.bases)

    @property
    @override
    def space_dim(self) -> int:
        return int(np.prod([space.space_dim for space in self.bases]))

    @override
    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
                f"spatial_dim={self.spatial_dim}, space_dim={self.space_dim}, "
                f"bases={self.bases!r}"
                ")")


@space_for_shape.register(TensorProductShape)
def space_for_tensor_product_shape(
        shape: TensorProductShape,
        order: int | tuple[int, ...]) -> TensorProductSpace:
    nbases = len(shape.bases)
    if isinstance(order, Number):
        assert isinstance(order, int)
        orders = (order,) * nbases
    else:
        assert isinstance(order, tuple)
        if len(order) != nbases:
            raise ValueError("must provide one order per base shape in 'shape'; "
                    f"got {order} for {len(shape.bases)} tensor product shapes")

        orders = order

    return TensorProductSpace(
          tuple(space_for_shape(shape.bases[i], orders[i]) for i in range(nbases)))

# }}}


# {{{ PN

class PN(FunctionSpace):
    r"""The function space of polynomials with total degree
    :math:`N` = :attr:`order`.

    .. math::

        P^N:=\operatorname{span}\left\{\prod_{i=1}^d x_i^{n_i}:\sum n_i\le N\right\}.
    """

    def __init__(self, spatial_dim: int, order: int) -> None:
        super().__init__()

        self._order: int = order
        self._spatial_dim: int = spatial_dim

    @property
    @override
    def order(self) -> int:
        """Total degree of the polynomials in the space."""
        return self._order

    @property
    @override
    def spatial_dim(self) -> int:
        return self._spatial_dim

    @property
    @override
    def space_dim(self) -> int:
        from math import comb
        return comb(self.order + self.spatial_dim, self.spatial_dim)

    @override
    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
                f"spatial_dim={self.spatial_dim}, order={self.order}"
                ")")


@space_for_shape.register(Simplex)
def space_for_simplex(shape: Simplex, order: int | tuple[int, ...]) -> PN:
    assert isinstance(order, int)
    return PN(shape.dim, order)

# }}}


# {{{ QN

class QN(TensorProductSpace):
    r"""The function space of polynomials with maximum degree
    :math:`N` = :attr:`order`:

    .. math::

        Q^N:=\operatorname{span}
        \left \{\prod_{i=1}^d x_i^{n_i}:\max n_i\le N\right\}.
    """

    @overload
    # pyright-ignore: they overlap, can't be helped.
    def __new__(cls, spatial_dim: Literal[1], order: int) -> PN: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __new__(cls, spatial_dim: int, order: int) -> QN: ...

    def __new__(cls, spatial_dim: int, order: int) -> FunctionSpace:
        if spatial_dim == 1:
            return PN(spatial_dim, order)
        else:
            return FunctionSpace.__new__(cls)

    def __init__(self, spatial_dim: int, order: int) -> None:
        super().__init__((PN(1, order),) * spatial_dim)

    @property
    @override
    def order(self):
        """Maximum degree of the polynomials in the space."""
        return self.bases[0].order

    @override
    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
                f"spatial_dim={self.spatial_dim}, order={self.order}"
                ")")

# }}}

# vim: foldmethod=marker
