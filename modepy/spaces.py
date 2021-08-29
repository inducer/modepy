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

from functools import singledispatch
from numbers import Number
from typing import Union, Tuple

import numpy as np

from modepy.shapes import Shape, Simplex, Hypercube


# {{{ function spaces

class FunctionSpace:
    r"""An opaque object representing a finite-dimensional function space
    of functions :math:`\mathbb{R}^n \to \mathbb{R}`.

    .. attribute:: spatial_dim

        :math:`n` in the above definition, the number of spatial dimensions
        in which the functions in the space operate.

    .. attribute:: space_dim

        The number of dimensions of the function space.
    """


@singledispatch
def space_for_shape(shape: Shape, order: int) -> FunctionSpace:
    r"""Return an unspecified instance of :class:`FunctionSpace` suitable
    for approximation on *shape* attaining interpolation error of
    :math:`O(h^{\text{order}+1})`.
    """
    raise NotImplementedError(type(shape).__name__)

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
        self.spatial_dim = spatial_dim
        self.order = order

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
def _space_for_simplex(shape: Simplex, order: int):
    return PN(shape.dim, order)

# }}}


# {{{ QN

class QN(FunctionSpace):
    r"""The function space of polynomials with maximum degree
    :math:`N` = :attr:`order`:

    .. math::

        Q^N:=\operatorname{span}
        \left \{\prod_{i=1}^d x_i^{n_i}:\max n_i\le N\right\}.

    .. attribute:: order

    .. automethod:: __init__
    """

    def __init__(self, spatial_dim: int, order: int) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.order = order

    @property
    def space_dim(self) -> int:
        return (self.order + 1)**self.spatial_dim

    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
            f"spatial_dim={self.spatial_dim}, order={self.order}"
            ")")


@space_for_shape.register(Hypercube)
def _space_for_hypercube(shape: Hypercube, order: Union[int, Tuple[int, ...]]):
    if isinstance(order, Number):
        return QN(shape.dim, order)
    else:
        return TensorProductSpace([QN(1, n) for n in order])

# }}}


# # {{{ generic tensor product space

class TensorProductSpace(FunctionSpace):
    """
    .. attribute:: bases

        A :class:`tuple` of the base spaces that take part in the
        tensor product.

    .. attribute:: order

        A :class:`tuple` of orders per spatial dimension.

    .. automethod:: __init__
    """

    def __init__(self, bases: Tuple[FunctionSpace, ...]) -> None:
        self.bases = bases

    @property
    def order(self):
        return tuple([space.order for space in self.bases])

    @property
    def spatial_dim(self) -> int:
        return sum(space.spatial_dim for space in self.bases)

    @property
    def space_dim(self) -> int:
        return np.prod([space.space_dim for space in self.bases])

# }}}

# vim: foldmethod=marker
