"""
.. currentmodule:: modepy

Function Spaces
---------------

.. autoclass:: FunctionSpace
.. autoclass:: PN
.. autoclass:: QN
.. autoclass:: TensorProductSpace
.. autofunction:: space_for_shape
"""

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

from abc import ABC, abstractproperty
from functools import singledispatch
from numbers import Number
from typing import Any, Tuple, Union

import numpy as np

from modepy.shapes import Shape, Simplex, Hypercube, TensorProductShape


# {{{ function spaces

class FunctionSpace(ABC):
    r"""An opaque object representing a finite-dimensional function space
    of functions :math:`\mathbb{R}^n \to \mathbb{R}`.

    .. attribute:: spatial_dim

        :math:`n` in the above definition, the number of spatial dimensions
        in which the functions in the space operate.

    .. attribute:: space_dim

        The number of dimensions of the function space.
    """

    @abstractproperty
    def spatial_dim(self) -> int:
        pass

    @abstractproperty
    def space_dim(self) -> int:
        pass


@singledispatch
def space_for_shape(
        shape: Shape, order: Union[int, Tuple[int, ...]]
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
    """
    .. attribute:: bases

        A :class:`tuple` of the base spaces that take part in the
        tensor product.

    .. automethod:: __init__
    """

    bases: Tuple[FunctionSpace, ...]

    # NOTE: https://github.com/python/mypy/issues/1020
    def __new__(cls, bases: Tuple[FunctionSpace, ...]) -> Any:
        if len(bases) == 1:
            return bases[0]
        else:
            return FunctionSpace.__new__(cls)

    def __init__(self, bases: Tuple[FunctionSpace, ...]) -> None:
        self.bases = sum([
            space.bases if isinstance(space, TensorProductSpace) else (space,)
            for space in bases
            ], ())

    @property
    def spatial_dim(self) -> int:
        return sum(space.spatial_dim for space in self.bases)

    @property
    def space_dim(self) -> int:
        return np.prod([space.space_dim for space in self.bases])

    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
                f"spatial_dim={self.spatial_dim}, space_dim={self.space_dim}, "
                f"bases={self.bases!r}"
                ")")


@space_for_shape.register(TensorProductShape)
def _space_for_tensor_product_shape(
        shape: TensorProductShape,
        order: Union[int, Tuple[int, ...]]) -> TensorProductSpace:
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

    return TensorProductSpace(tuple([
        space_for_shape(shape.bases[i], orders[i]) for i in range(nbases)
        ]))

# }}}


# {{{ PN

class PN(FunctionSpace):
    r"""The function space of polynomials with total degree
    :math:`N` = :attr:`order`.

    .. math::

        P^N:=\operatorname{span}\left\{\prod_{i=1}^d x_i^{n_i}:\sum n_i\le N\right\}.

    .. attribute:: order

    .. automethod:: __init__
    """

    def __init__(self, spatial_dim: int, order: int) -> None:
        super().__init__()
        self._spatial_dim = spatial_dim
        self.order = order

    @property
    def spatial_dim(self) -> int:
        return self._spatial_dim

    @property
    def space_dim(self) -> int:
        spdim = self.spatial_dim
        order = self.order
        try:
            from math import comb       # comb is v3.8+
            return comb(order + spdim, spdim)
        except ImportError:
            from functools import reduce
            from operator import mul
            return reduce(mul, range(order + 1, order + spdim + 1), 1) \
                    // reduce(mul, range(1, spdim + 1), 1)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
                f"spatial_dim={self.spatial_dim}, order={self.order}"
                ")")


@space_for_shape.register(Simplex)
def _space_for_simplex(shape: Simplex, order: Union[int, Tuple[int, ...]]) -> PN:
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

    .. attribute:: order

    .. automethod:: __init__
    """

    # NOTE: https://github.com/python/mypy/issues/1020
    def __new__(cls, spatial_dim: int, order: int) -> Any:
        if spatial_dim == 1:
            return PN(spatial_dim, order)
        else:
            return FunctionSpace.__new__(cls)

    def __init__(self, spatial_dim: int, order: int) -> None:
        super().__init__((PN(1, order),) * spatial_dim)

    @property
    def order(self):
        return self.bases[0].order

    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
                f"spatial_dim={self.spatial_dim}, order={self.order}"
                ")")


@space_for_shape.register(Hypercube)
def _space_for_hypercube(
        shape: Hypercube, order: Union[int, Tuple[int, ...]]
        ) -> TensorProductSpace:
    if isinstance(order, Number):
        assert isinstance(order, int)
        return QN(shape.dim, order)
    else:
        assert isinstance(order, tuple)
        if len(order) != shape.dim:
            raise ValueError("must provide one order per dimension; "
                    f"got {order} for a {shape.dim}d hypercube")

        return TensorProductSpace(tuple([PN(1, n) for n in order]))

# }}}

# vim: foldmethod=marker
