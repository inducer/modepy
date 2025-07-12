# {{{ docstring

r"""
:mod:`modepy.shapes` provides a generic description of the supported shapes
(i.e. reference elements).

.. currentmodule:: modepy

.. autoclass:: Shape
    :members:
.. autoclass:: Face
    :members:

.. autofunction:: unit_vertices_for_shape
.. autofunction:: faces_for_shape
.. autofunction:: face_normal

Tensor Product Shapes
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: TensorProductShape
    :members:
    :show-inheritance:

Simplices
^^^^^^^^^

.. autoclass:: Simplex
    :members:
    :show-inheritance:

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
    :members:
    :show-inheritance:

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

Bi-unit coordinates on :math:`(r, s, t)` (also called 'unit' coordinates)::

    t
    ^
    |
    E----------G
    |\         |\
    | \        | \
    |  \       |  \
    |   F------+---H
    |   |  O   |   |
    A---+------C---|--> s
     \  |       \  |
      \ |        \ |
       \|         \|
        B----------D
         \
          v r

Vertices in bi-unit coordinates::

    O = ( 0,  0,  0)
    A = (-1, -1, -1)
    B = ( 1, -1, -1)
    C = (-1,  1, -1)
    D = ( 1,  1, -1)
    E = (-1, -1,  1)
    F = ( 1, -1,  1)
    G = (-1,  1,  1)
    H = ( 1,  1,  1)

The order of the vertices in the hypercubes follows binary counting
in ``tsr`` (i.e. in reverse axis order).
For example, in 3D, ``A, B, C, D, ...`` is ``000, 001, 010, 011, ...``.

Submeshes
---------
.. autofunction:: submesh_for_shape
"""

# }}}
from __future__ import annotations


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

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, singledispatch
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from modepy.typing import ArrayF


# {{{ interface

@dataclass(frozen=True)
class Shape(ABC):
    dim: int
    """Spatial dimension of the shape."""

    @property
    @abstractmethod
    def nvertices(self) -> int:
        """Number of vertices."""

    @property
    @abstractmethod
    def nfaces(self) -> int:
        """Number of faces."""


@singledispatch
def unit_vertices_for_shape(shape: Shape) -> ArrayF:
    """
    :returns: an :class:`~numpy.ndarray` of shape `(dim, nvertices)`.
    """
    raise NotImplementedError(type(shape).__name__)


@dataclass(frozen=True)
class Face:
    """Mix-in to be used with a concrete :class:`Shape` subclass to represent
    geometry information about a face of a shape.
    """

    volume_shape: Shape
    """The volume :class:`Shape` to which the face belongs."""
    face_index: int
    """The face index in :attr:`volume_shape` of this face."""
    volume_vertex_indices: tuple[int, ...]
    """A tuple of indices into the vertices returned by
    :func:`unit_vertices_for_shape` for the :attr:`volume_shape`.
    """
    map_to_volume: Callable[[ArrayF], ArrayF]
    """A :class:`~collections.abc.Callable` that takes an array of
    size `(dim, nnodes)` of unit nodes on the face represented by
    *face_vertices* and maps them to the :attr:`volume_shape`.
    """


def face_normal(face: Face, normalize: bool = True) -> ArrayF:
    """
    .. versionadded :: 2021.2.1
    """
    volume_vertices = unit_vertices_for_shape(face.volume_shape)
    face_vertices = volume_vertices[:, face.volume_vertex_indices]

    assert isinstance(face, Shape)

    if face.dim == 0:
        # FIXME Grrrr. Hardcoded special case. Got a better idea?
        (fv,), = face_vertices
        return np.array([np.sign(fv)])

    # Compute the outer product of the vectors spanning the surface, obtaining
    # the surface pseudoscalar.
    from functools import reduce
    from operator import xor as outerprod

    from pymbolic.geometric_algebra import MultiVector
    surface_ps: MultiVector[np.floating] = reduce(outerprod, [
        MultiVector(face_vertices[:, i+1] - face_vertices[:, 0])
        for i in range(face.dim)])

    if normalize:
        surface_ps = surface_ps / np.sqrt(surface_ps.norm_squared())

    # Compute the normal as the dual of the surface pseudoscalar.
    return surface_ps.dual().as_vector()


@singledispatch
def faces_for_shape(shape: Shape) -> tuple[Face, ...]:
    r"""
    :returns: a tuple of :class:`Face`\ s representing the faces of *shape*.
    """
    raise NotImplementedError(type(shape).__name__)

# }}}


# {{{ tensor product shape

@dataclass(frozen=True, init=False)
class TensorProductShape(Shape):
    """A shape formed as a tensor product of other shapes.

    For example, the tensor product of a line segment and a triangle (2D simplex)
    results in a prism shape. Special cases include the :class:`Hypercube`.
    """

    bases: tuple[Shape, ...]
    """A :class:`tuple` of base shapes that form the tensor product."""

    def __new__(cls, bases: tuple[Shape, ...]) -> Any:
        if len(bases) == 1:
            return bases[0]
        else:
            return Shape.__new__(cls)

    def __init__(self, bases: tuple[Shape, ...]) -> None:
        # flatten input shapes
        bases = sum((
            s.bases if isinstance(s, TensorProductShape) else (s,)
            for s in bases
            ), ())

        nsegments = len([s for s in bases if s.dim == 1])
        if nsegments < len(bases) - 1:
            raise NotImplementedError(
                    "only tensor products between (at most) one arbitrary polygonal "
                    f"shape and line segments supported; got {nsegments} line "
                    f"segments in a total of {len(bases)} shapes")

        dim = sum(s.dim for s in bases)
        super().__init__(dim)
        object.__setattr__(self, "bases", bases)

    @property
    def nvertices(self) -> int:
        return int(np.prod([s.nvertices for s in self.bases]))

    @property
    def nfaces(self) -> int:
        # FIXME: this obviously only works for `shape x segment x segment x ...`
        *segments, shape = sorted(self.bases, key=lambda s: s.dim)
        return shape.nfaces + len(segments) * segments[0].nfaces


@unit_vertices_for_shape.register(TensorProductShape)
def _unit_vertices_for_tp(shape: TensorProductShape):
    from modepy.nodes import tensor_product_nodes
    return tensor_product_nodes([
        unit_vertices_for_shape(s) for s in shape.bases
        ])

# }}}


# {{{ simplex

@dataclass(frozen=True)
class Simplex(Shape):
    """An n-dimensional simplex (lines, triangle, tetrahedron, etc.)."""

    @property
    def nfaces(self) -> int:
        return self.dim + 1

    @property
    def nvertices(self) -> int:
        return self.dim + 1


@dataclass(frozen=True)
class _SimplexFace(Simplex, Face):
    pass


@unit_vertices_for_shape.register(Simplex)
def _unit_vertices_for_simplex(shape: Simplex):
    result = np.empty((shape.dim, shape.dim+1), np.float64)
    result.fill(-1)

    for i in range(shape.dim):
        result[i, i+1] = 1

    return result


def _simplex_face_to_vol_map(
            face_vertices: ArrayF,
            p: NDArray[np.integer]
        ) -> ArrayF:
    dim, npoints = face_vertices.shape
    if npoints != dim:
        raise ValueError("'face_vertices' has wrong shape")

    origin = face_vertices[:, 0].reshape(-1, 1)
    face_basis = face_vertices[:, 1:] - origin

    return origin + np.einsum("ij,jk->ik", face_basis, (1 + p) / 2)


_SIMPLEX_FACES: dict[int, tuple[tuple[int, ...], ...]] = {
            1: ((0,), (1,)),
            2: ((0, 1), (2, 0), (1, 2)),
            3: ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
            }


@faces_for_shape.register(Simplex)
def _faces_for_simplex(shape: Simplex):
    # NOTE: order is chosen to maintain a positive orientation
    face_vertex_indices = _SIMPLEX_FACES[shape.dim]

    vertices = unit_vertices_for_shape(shape)
    return [
            _SimplexFace(
                dim=shape.dim-1,
                volume_shape=shape, face_index=iface,
                volume_vertex_indices=fvi,
                map_to_volume=partial(_simplex_face_to_vol_map, vertices[:, fvi]))
            for iface, fvi in enumerate(face_vertex_indices)]

# }}}


# {{{ hypercube

@dataclass(frozen=True)
class Hypercube(TensorProductShape):
    """An n-dimensional hypercube (line, square, hexahedron, etc.)."""

    def __new__(cls, dim: int) -> Any:
        if dim == 1:
            return Simplex(1)
        else:
            return Shape.__new__(cls)

    def __init__(self, dim: int) -> None:
        super().__init__((Simplex(1),) * dim)

    def __getnewargs__(self):
        # NOTE: ensures Hypercube is picklable
        return (self.dim,)


@dataclass(frozen=True, init=False)
class _HypercubeFace(Hypercube, Face):
    def __new__(cls, dim, **kwargs) -> Any:
        if dim == 1:
            return _SimplexFace(dim=1, **kwargs)
        else:
            return Shape.__new__(cls)

    def __init__(self, **kwargs) -> None:
        Hypercube.__init__(self, kwargs.pop("dim"))
        Face.__init__(self, **kwargs)


def _hypercube_face_to_vol_map(
            face_vertices: ArrayF,
            p: NDArray[np.integer]
        ) -> ArrayF:
    dim, npoints = face_vertices.shape
    if npoints != 2**(dim - 1):
        raise ValueError("'face_vertices' has wrong shape")

    origin = face_vertices[:, 0].reshape(-1, 1)

    if dim <= 3:
        # works up to (and including) 3D:
        # - no-op for 1D, 2D
        # - For square faces, eliminate node opposite origin
        face_basis = face_vertices[:, 1:3] - origin
    else:
        raise NotImplementedError(f"_hypercube_face_to_vol_map in {dim} dimensions")

    return origin + np.einsum("ij,jk->ik", face_basis, (1 + p) / 2)


_HYPERCUBE_FACES: dict[int, tuple[tuple[int, ...], ...]] = {
        1: ((0b0,), (0b1,)),
        2: ((0b00, 0b01), (0b11, 0b10), (0b10, 0b00), (0b01, 0b11)),
        3: (
            (0b000, 0b010, 0b001, 0b011,),
            (0b100, 0b101, 0b110, 0b111,),

            (0b000, 0b100, 0b010, 0b110,),
            (0b001, 0b011, 0b101, 0b111,),

            (0b000, 0b001, 0b100, 0b101,),
            (0b010, 0b110, 0b011, 0b111,),
            )
        }


@faces_for_shape.register(Hypercube)
def _faces_for_hypercube(shape: Hypercube):
    # NOTE: order is chosen to maintain a positive orientation
    face_vertex_indices = _HYPERCUBE_FACES[shape.dim]

    vertices = unit_vertices_for_shape(shape)
    return [
            _HypercubeFace(
                dim=shape.dim-1,
                volume_shape=shape, face_index=iface,
                volume_vertex_indices=fvi,
                map_to_volume=partial(_hypercube_face_to_vol_map, vertices[:, fvi]))
            for iface, fvi in enumerate(face_vertex_indices)]

# }}}


# {{{ submeshes

@singledispatch
def submesh_for_shape(
        shape: Shape, node_tuples: Sequence[tuple[int, ...]]
        ) -> Sequence[tuple[int, ...]]:
    """Return a list of tuples of indices into the node list that
    generate a tessellation of the reference element.

    :arg node_tuples: A list of tuples *(i, j, ...)* of integers
        indicating node positions inside the unit element. The
        returned list references indices in this list.

        :func:`modepy.node_tuples_for_space` may be used to generate *node_tuples*.

    .. versionadded:: 2020.3
    """
    raise NotImplementedError(type(shape).__name__)


@submesh_for_shape.register(Simplex)
def _submesh_for_simplex(shape: Simplex, node_tuples):
    from pytools import add_tuples, single_valued
    dims = single_valued(len(nt) for nt in node_tuples)

    node_dict = {
            ituple: idx
            for idx, ituple in enumerate(node_tuples)}

    if dims == 1:
        result = []

        def try_add_line(d1, d2):
            with contextlib.suppress(KeyError):
                result.append((
                    node_dict[add_tuples(current, d1)],
                    node_dict[add_tuples(current, d2)],
                    ))

        # https://github.com/PyCQA/flake8-bugbear/issues/175
        for current in node_tuples:  # noqa: B007
            try_add_line((0,), (1,),)

        return result
    elif dims == 2:
        # {{{ triangle sub-mesh
        result = []

        def try_add_tri(d1, d2, d3):
            with contextlib.suppress(KeyError):
                result.append((
                    node_dict[add_tuples(current, d1)],
                    node_dict[add_tuples(current, d2)],
                    node_dict[add_tuples(current, d3)],
                    ))

        # https://github.com/PyCQA/flake8-bugbear/issues/175
        for current in node_tuples:  # noqa: B007
            # this is a tessellation of a square into two triangles.
            # subtriangles that fall outside of the master triangle are
            # simply not added.

            # positively oriented
            try_add_tri((0, 0), (1, 0), (0, 1))
            try_add_tri((1, 0), (1, 1), (0, 1))

        return result

        # }}}
    elif dims == 3:
        # {{{ tet sub-mesh

        def try_add_tet(d1, d2, d3, d4):
            with contextlib.suppress(KeyError):
                result.append((
                    node_dict[add_tuples(current, d1)],
                    node_dict[add_tuples(current, d2)],
                    node_dict[add_tuples(current, d3)],
                    node_dict[add_tuples(current, d4)],
                    ))

        result = []
        # https://github.com/PyCQA/flake8-bugbear/issues/175
        for current in node_tuples:  # noqa: B007
            # this is a tessellation of a cube into six tets.
            # subtets that fall outside of the master tet are simply not added.

            # positively oriented
            try_add_tet((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
            try_add_tet((1, 0, 1), (1, 0, 0), (0, 0, 1), (0, 1, 0))
            try_add_tet((1, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 1))

            try_add_tet((1, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 0))
            try_add_tet((0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 1))
            try_add_tet((0, 1, 1), (1, 1, 1), (1, 0, 1), (1, 1, 0))

        return result

        # }}}
    else:
        raise NotImplementedError(f"{dims}-dimensional sub-meshes")


@submesh_for_shape.register(TensorProductShape)
def _submesh_for_hypercube(shape: TensorProductShape, node_tuples):
    from modepy.spaces import space_for_shape
    space = space_for_shape(shape, order=1)
    from modepy.nodes import node_tuples_for_space
    vertex_node_tuples = node_tuples_for_space(space)

    from pytools import add_tuples
    result = []
    node_dict = {ituple: idx for idx, ituple in enumerate(node_tuples)}
    for current in node_tuples:
        with contextlib.suppress(KeyError):
            result.append(tuple(
                    node_dict[add_tuples(current, offset)]
                    for offset in vertex_node_tuples
                    ))

    return result

# }}}


# vim: foldmethod=marker
