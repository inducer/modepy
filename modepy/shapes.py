r"""
:mod:`modepy.shapes` provides a generic description of the supported shapes
(i.e. reference elements).

.. autoclass:: Shape
.. autofunction:: biunit_vertices_for_shape
.. autofunction:: face_vertex_indices_for_shape
.. autofunction:: face_map_for_shape

Simplices
^^^^^^^^^

.. autoclass:: Simplex

.. _tri-coords:

Coordinates on the triangle
---------------------------

Bi-unit coordinates :math:`(r, s)` (also called 'unit' coordinates)::

    ^ s
    |
    C
    |\
    | \
    |  O
    |   \
    |    \
    A-----B--> r

Vertices in bi-unit coordinates::

    O = ( 0,  0)
    A = (-1, -1)
    B = ( 1, -1)
    C = (-1,  1)

Equilateral coordinates :math:`(x, y)`::

          C
         / \
        /   \
       /     \
      /   O   \
     /         \
    A-----------B

Vertices in equilateral coordinates::

    O = ( 0,          0)
    A = (-1, -1/sqrt(3))
    B = ( 1, -1/sqrt(3))
    C = ( 0,  2/sqrt(3))

.. _tet-coords:

Coordinates on the tetrahedron
------------------------------

Bi-unit coordinates :math:`(r, s, t)` (also called 'unit' coordinates)::

               ^ s
               |
               C
              /|\
             / | \
            /  |  \
           /   |   \
          /   O|    \
         /   __A-----B---> r
        /_--^ ___--^^
       ,D--^^^
    t L

(squint, and it might start making sense...)

Vertices in bi-unit coordinates :math:`(r, s, t)`::

    O = ( 0,  0,  0)
    A = (-1, -1, -1)
    B = ( 1, -1, -1)
    C = (-1,  1, -1)
    D = (-1, -1,  1)

Vertices in equilateral coordinates :math:`(x, y, z)`::

    O = ( 0,          0,          0)
    A = (-1, -1/sqrt(3), -1/sqrt(6))
    B = ( 1, -1/sqrt(3), -1/sqrt(6))
    C = ( 0,  2/sqrt(3), -1/sqrt(6))
    D = ( 0,          0,  3/sqrt(6))

Hypercubes
^^^^^^^^^^

.. autoclass:: Hypercube

.. _square-coords:

Coordinates on the square
-------------------------

Bi-unit coordinates on :math:`(r, s)` (also called 'unit' coordinates)::

     ^ s
     |
     C---------D
     |         |
     |         |
     |    O    |
     |         |
     |         |
     A---------B --> r


Vertices in bi-unit coordinates::

    O = ( 0,  0)
    A = (-1, -1)
    B = ( 1, -1)
    C = (-1,  1)
    D = ( 1,  1)

.. _cube-coords:

Coordinates on the cube
-----------------------

Unit coordinates on :math:`(r, s, t)` (also called 'unit' coordinates)::

    t
    ^
    |
    B----------D
    |\         |\
    | \        | \
    |  \       |  \
    |   F------+---H
    |   |  O   |   |
    A---+------C---|--> s
     \  |       \  |
      \ |        \ |
       \|         \|
        E----------G
         \
          v r

Vertices in unit coordinates::

    O = ( 0,  0,  0)
    A = (-1, -1, -1)
    B = (-1, -1,  1)
    C = (-1,  1, -1)
    D = (-1,  1,  1)
    E = ( 1, -1, -1)
    F = ( 1, -1,  1)
    G = ( 1,  1, -1)
    H = ( 1,  1,  1)

The order of the vertices in the hypercubes follows binary counting
in ``rst``. For example, in 3D, ``A, B, C, D, ...`` is ``000, 001, 010, 011, ...``.
"""


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
    .. attribute :: dim
    .. attribute :: nfaces
    .. attribute :: nvertices
    """
    dim: int


@singledispatch
def biunit_vertices_for_shape(shape: Shape):
    """
    :returns: a :class:`~numpy.ndarray` of shape `(dim, nvertices)`.
    """
    raise NotImplementedError(type(shape).__name__)


@singledispatch
def face_vertex_indices_for_shape(shape: Shape):
    """
    :results: a tuple of the length :attr:`Shape.nfaces`, where each entry
        is a tuple of indices into the vertices returned by
        :func:`biunit_vertices_for_shape`.
    """
    raise NotImplementedError(type(shape).__name__)


@singledispatch
def face_map_for_shape(shape: Shape, face_vertices: np.ndarray):
    """
    :returns: a :class:`~collections.abc.Callable` that takes an array of
        size `(dim, nnodes)` of unit nodes on the face represented by
        *face_vertices* and maps them to the volume.
    """
    raise NotImplementedError(type(shape).__name__)

# }}}


# {{{ simplex

class Simplex(Shape):
    @property
    def nfaces(self):
        return self.dim + 1

    @property
    def nvertices(self):
        return self.dim + 1


@biunit_vertices_for_shape.register
def _(shape: Simplex):
    from modepy.tools import unit_vertices
    return unit_vertices(shape.dim).T.copy()


@face_vertex_indices_for_shape.register
def _(shape: Simplex):
    fvi = np.empty((shape.dim + 1, shape.dim), dtype=np.int)
    indices = np.arange(shape.dim + 1)

    for iface in range(shape.nfaces):
        fvi[iface, :] = np.hstack([indices[:iface], indices[iface + 1:]])

    return fvi


@face_map_for_shape.register
def _(shape: Simplex, face_vertices: np.ndarray):
    dim, npoints = face_vertices.shape
    if npoints != dim:
        raise ValueError("'face_vertices' has wrong shape")

    origin = face_vertices[:, 0].reshape(-1, 1)
    face_basis = face_vertices[:, 1:] - origin

    return lambda p: origin + np.einsum("ij,jk->ik", face_basis, (1 + p) / 2)

# }}}


# {{{ hypercube

class Hypercube(Shape):
    @property
    def nfaces(self):
        return 2 * self.dim

    @property
    def nvertices(self):
        return 2**self.dim


@biunit_vertices_for_shape.register
def _(shape: Hypercube):
    from modepy.nodes import tensor_product_nodes
    return tensor_product_nodes(shape.dim, np.array([-1.0, 1.0]))


@face_vertex_indices_for_shape.register
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
        }[shape.dim]


@face_map_for_shape.register
def _(shape: Hypercube, face_vertices: np.ndarray):
    dim, npoints = face_vertices.shape
    if npoints != 2**(dim - 1):
        raise ValueError("'face_vertices' has wrong shape")

    origin = face_vertices[:, 0].reshape(-1, 1)
    face_basis = face_vertices[:, -2:0:-1] - origin

    return lambda p: origin + np.einsum("ij,jk->ik", face_basis, (1 + p) / 2)

# }}}

# vim: foldmethod=marker
