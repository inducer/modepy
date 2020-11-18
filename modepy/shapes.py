__copyright__ = """
Copyright (c) 2013 Andreas Kloeckner
Copyright (c) 2020 Alexandru Fikl
"""

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

from functools import singledispatch
from dataclasses import dataclass


# {{{ interface

@dataclass(frozen=True)
class Shape:
    """
    .. attribute :: dims
    .. attribute :: nfaces
    .. attribute :: nvertices
    """
    dims: int


@singledispatch
def get_unit_vertices(shape: Shape):
    """
    :returns: an :class:`~numpy.ndarray` of shape `(nvertices, dims)`.
    """
    raise NotImplementedError(type(shape).__name__)


@singledispatch
def get_face_vertex_indices(shape: Shape):
    """
    :results: a tuple of the length :attr:`Shape.nfaces`, where each entry
        is a tuple of indices into the vertices returned by
        :func:`get_unit_vertices`.
    """
    raise NotImplementedError(type(shape).__name__)


@singledispatch
def get_face_map(shape: Shape, face_vertices: np.ndarray):
    """
    :returns: a :class:`~collections.abc.Callable` that takes an array of
        size `(dims, nnodes)` of unit nodes on the face represented by
        *face_vertices* and maps them to the volume.
    """
    raise NotImplementedError(type(shape).__name__)


@singledispatch
def get_quadrature(shape: Shape, order: int):
    """
    :returns: a :class:`~modepy.Quadrature` that is exact up to :math:`2 N + 1`.
    """
    raise NotImplementedError(type(shape).__name__)


@singledispatch
def get_node_count(shape: Shape, order: int):
    raise NotImplementedError(type(shape).__name__)


@singledispatch
def get_node_tuples(shape: Shape, order: int):
    raise NotImplementedError(type(shape).__name__)


@singledispatch
def get_unit_nodes(shape: Shape, order: int):
    raise NotImplementedError(type(shape).__name__)


@singledispatch
def get_basis(shape: Shape, order: int):
    raise NotImplementedError(type(shape).__name__)


@singledispatch
def get_grad_basis(shape: Shape, order: int):
    raise NotImplementedError(type(shape).__name__)


@singledispatch
def get_basis_with_mode_ids(shape: Shape, order: int):
    raise NotImplementedError(type(shape).__name__)

# }}}


# {{{ simplex

class Simplex(Shape):
    @property
    def nfaces(self):
        return self.dims + 1

    @property
    def nvertices(self):
        return self.dim + 1


@get_unit_vertices.register(Simplex)
def _(shape: Simplex):
    from modepy.tools import unit_vertices
    return unit_vertices(shape.dims)


@get_face_vertex_indices.register(Simplex)
def _(shape: Simplex):
    fvi = np.empty((shape.dims + 1, shape.dims), dtype=np.int)
    indices = np.arange(shape.dims + 1)

    for iface in range(shape.nfaces):
        fvi[iface, :] = np.hstack([indices[:iface], indices[iface + 1:]])

    return fvi


@get_face_map.register(Simplex)
def _(shape: Simplex, face_vertices: np.ndarray):
    dims, npoints = face_vertices.shape
    if npoints != dims:
        raise ValueError("'face_vertices' has wrong shape")

    origin = face_vertices[:, 0].reshape(-1, 1)
    face_basis = face_vertices[:, 1:] - origin

    return lambda p: origin + np.einsum("ij,jk->ik", face_basis, (1 + p) / 2)


@get_quadrature.register(Simplex)
def _(shape: Simplex, order: int):
    import modepy as mp
    try:
        quad = mp.XiaoGimbutasSimplexQuadrature(2*order + 1, shape.dims)
    except (mp.QuadratureRuleUnavailable, ValueError):
        quad = mp.GrundmannMoellerSimplexQuadrature(order, shape.dims)

    return quad

# }}}


# {{{ hypercube

class Hypercube(Shape):
    @property
    def nfaces(self):
        return 2 * self.dims

    @property
    def nvertices(self):
        return 2**self.dims


@get_unit_vertices.register(Hypercube)
def _(shape: Hypercube):
    from modepy.nodes import tensor_product_nodes
    return tensor_product_nodes(shape.dims, np.array([-1.0, 1.0])).T


@get_face_vertex_indices.register(Hypercube)
def _(shape: Hypercube):
    # FIXME: replace by nicer n-dimensional formula
    return {
        1: ((0b0,), (0b1,)),
        2: ((0b00, 0b01), (0b10, 0b11), (0b00, 0b10), (0b01, 0b11)),
        3: (
            (0b000, 0b001, 0b010, 0b011,),
            (0b100, 0b101, 0b110, 0b111,),

            (0b000, 0b010, 0b100, 0b110,),
            (0b001, 0b011, 0b101, 0b111,),

            (0b000, 0b001, 0b100, 0b101,),
            (0b010, 0b011, 0b110, 0b111,),
            )
        }[shape.dims]


@get_face_map.register(Hypercube)
def _(shape: Hypercube, face_vertices: np.ndarray):
    dims, npoints = face_vertices.shape
    if npoints != 2**(dims - 1):
        raise ValueError("'face_vertices' has wrong shape")

    origin = face_vertices[:, 0].reshape(-1, 1)
    face_basis = face_vertices[:, -2:0:-1] - origin

    return lambda p: origin + np.einsum("ij,jk->ik", face_basis, (1 + p) / 2)


@get_quadrature.register(Hypercube)
def _(shape: Hypercube, order: int):
    import modepy as mp
    if shape.dims == 0:
        quad = mp.Quadrature(np.empty((0, 1)), np.empty((0, 1)))
    else:
        from modepy.quadrature import LegendreGaussTensorProductQuadrature
        quad = LegendreGaussTensorProductQuadrature(order, shape.dims)

    return quad

# }}}
