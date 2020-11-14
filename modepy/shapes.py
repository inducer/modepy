__copyright__ = """
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
from dataclasses import dataclass, field

__doc__ = """
Shapes
------

.. currentmodule:: modepy

.. autoclass:: Shape
.. autoclass:: Simplex
.. autoclass:: Hypercube

.. autofunction:: get_unit_vertices
.. autofunction:: get_face_map
.. autofunction:: get_quadrature
"""


# {{{ interface

@dataclass(frozen=True)
class Shape:
    dims: int
    nfaces: int = field(init=False)


@singledispatch
def get_unit_vertices(shape: Shape):
    """
    :returns: an :class:`~numpy.ndarray` of shape ``(nvertices, dims)``.
    """
    raise NotImplementedError


@singledispatch
def get_face_vertex_indices(shape: Shape):
    """
    :results: indices into the vertices returned by :func:`get_unit_vertices`
        belonging to each face.
    """

    raise NotImplementedError


@singledispatch
def get_face_map(shape: Shape, face_vertices: np.ndarray):
    """
    :returns: a :class:`~collections.abc.Callable` that takes an array of
        unit nodes on the face represented by *face_vertices* and maps
        them to the volume.
    """
    raise NotImplementedError


@singledispatch
def get_quadrature(shape: Shape, order: int):
    """
    :returns: a :class:`~modepy.Quadrature` instance of the given *order*.
    """
    raise NotImplementedError


@singledispatch
def get_unit_nodes(shape: Shape, order: int):
    raise NotImplementedError


@singledispatch
def get_basis(shape: Shape, order: int):
    raise NotImplementedError


@singledispatch
def get_grad_basis(shape: Shape, order: int):
    raise NotImplementedError

# }}}


# {{{ simplex

class Simplex(Shape):
    @property
    def nfaces(self):
        return self.dims + 1


@get_unit_vertices.register
def _(shape: Simplex):
    from modepy.tools import unit_vertices
    return unit_vertices(shape.dims)


@get_face_vertex_indices.register
def _(shape: Simplex):
    fvi = np.empty((shape.dims + 1, shape.dims), dtype=np.int)
    indices = np.arange(shape.dims + 1)

    for iface in range(shape.nfaces):
        fvi[iface, :] = np.hstack([indices[:iface], indices[iface + 1:]])

    return fvi


@get_face_map.register
def _(shape: Simplex, face_vertices: np.ndarray):
    dims, npoints = face_vertices.shape
    if npoints != dims:
        raise ValueError("'face_vertices' has wrong shape")

    origin = face_vertices[:, 0].reshape(-1, 1)
    face_basis = face_vertices[:, 1:] - origin

    return lambda p: origin + np.einsum("ij,jk->ik", face_basis, (1 + p) / 2)


@get_quadrature.register
def _(shape: Simplex, order: int):
    import modepy as mp
    if shape.dims == 0:
        quad = mp.Quadrature(np.empty((0, 1)), np.empty((0, 1)))
    else:
        try:
            quad = mp.XiaoGimbutasSimplexQuadrature(2*order + 1, shape.dims)
        except (mp.QuadratureRuleUnavailable, ValueError):
            quad = mp.GrundmannMoellerSimplexQuadrature(order, shape.dims)

    return quad


@get_unit_nodes.register
def _(shape: Simplex, order: int):
    import modepy as mp
    return mp.warp_and_blend_nodes(shape.dims, order)


@get_basis.register
def _(shape: Simplex, order: int):
    import modepy as mp
    return mp.simplex_onb(shape.dims, order)


@get_grad_basis.register
def _(shape: Simplex, order: int):
    import modepy as mp
    return mp.grad_simplex_onb(shape.dims, order)

# }}}


# {{{ hypercube

class Hypercube(Shape):
    @property
    def nfaces(self):
        return 2 * self.dims


@get_unit_vertices.register
def _(shape: Hypercube):
    from modepy.nodes import tensor_product_nodes
    return tensor_product_nodes(shape.dims, np.array([-1.0, 1.0])).T


@get_face_vertex_indices.register
def _(shape: Hypercube):
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


@get_face_map.register
def _(shape: Hypercube, face_vertices: np.ndarray):
    dims, npoints = face_vertices.shape
    if npoints != 2**(dims - 1):
        raise ValueError("'face_vertices' has wrong shape")

    origin = face_vertices[:, 0].reshape(-1, 1)
    face_basis = face_vertices[:, -2:0:-1] - origin

    return lambda p: origin + np.einsum("ij,jk->ik", face_basis, (1 + p) / 2)


@get_quadrature.register
def _(shape: Hypercube, order: int):
    import modepy as mp
    if shape.dims == 0:
        quad = mp.Quadrature(np.empty((0, 1)), np.empty((0, 1)))
    else:
        from modepy.quadrature import LegendreGaussTensorProductQuadrature
        quad = LegendreGaussTensorProductQuadrature(order, shape.dims)

    return quad


@get_unit_nodes.register
def _(shape: Hypercube, order: int):
    import modepy as mp
    return mp.legendre_gauss_lobatto_tensor_product_nodes(shape.dims, order)


@get_basis.register
def _(shape: Hypercube, order: int):
    import modepy as mp
    return mp.legendre_tensor_product_basis(shape.dims, order)


@get_grad_basis.register
def _(shape: Hypercube, order: int):
    import modepy as mp
    return mp.grad_legendre_tensor_product_basis(shape.dims, order)

# }}}
