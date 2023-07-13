"""
Transformations between coordinate systems on the simplex
---------------------------------------------------------

All of these expect and return arrays of shape *(dims, npts)*.

.. autofunction:: equilateral_to_unit
.. autofunction:: barycentric_to_unit
.. autofunction:: unit_to_barycentric
.. autofunction:: barycentric_to_equilateral

Interpolation quality
---------------------

.. autofunction:: estimate_lebesgue_constant

DOF arrays of tensor product discretizations
--------------------------------------------

.. autofunction:: reshape_array_for_tensor_product_space
.. autofunction:: unreshape_array_for_tensor_product_space

Types used in documentation
---------------------------

.. class:: ReshapeableT

    An array-like protocol that supports finding the shape and reshaping.

    .. attribute::: shape
    .. method::: reshape

"""

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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

from typing import Tuple, TypeVar
from typing import Protocol, runtime_checkable

from functools import reduce
from math import gamma      # noqa: F401
from math import sqrt

import numpy as np
import numpy.linalg as la

from pytools import memoize_method, MovedFunctionDeprecationWrapper
import modepy.shapes as shp
from modepy.spaces import TensorProductSpace


class Monomial:
    r"""A monomial

    .. math::

        \alpha \prod_{i=1}^d \xi_i^{e_i}

    where :math:`e` is the vector *exponents*,
    :math:`\alpha` is the scalar *factor*,
    and :math:`xi` is zero at :math:`(-1,\dots,-1)`
    and and one at :math:`(1,\dots,1)`.
    """
    def __init__(self, exponents, factor=1):
        self.exponents = exponents
        self.ones = np.ones((len(self.exponents),))
        self.factor = factor

    def __call__(self, xi):
        """Evaluate the monomial at *xi*.

        :arg: *xi* has shape *(d, ...)*.
        """
        from operator import mul

        x = (xi+1)/2
        return self.factor * \
                reduce(mul, (x[i]**expn
                    for i, expn in enumerate(self.exponents)))

    def simplex_integral(self):
        r"""Integral over the simplex
        :math:`\{\mathbf{x} \in [0, 1]^n: \sum x_i \le 1 \}`."""
        import math
        from operator import mul

        return (self.factor * 2**len(self.exponents)
                * reduce(mul, (math.factorial(alpha) for alpha in self.exponents))
                / math.factorial(len(self.exponents) + sum(self.exponents)))

    def hypercube_integral(self):
        """Integral over the hypercube :math:`[0, 1]^n`."""
        from functools import reduce
        return reduce(
                lambda integral, n: integral * 1 / (n + 1),
                self.exponents, 1.0)

    def diff(self, coordinate):
        diff_exp = list(self.exponents)
        orig_exp = diff_exp[coordinate]
        if orig_exp == 0:
            return Monomial(diff_exp, 0)
        diff_exp[coordinate] = orig_exp-1
        return Monomial(diff_exp, self.factor*orig_exp)


# {{{ coordinate mapping

class AffineMap:
    def __init__(self, a, b):
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)

    def __call__(self, x):
        """Apply the map *self* to a batch of vectors *x*.

        :arg x: has shape *(d, npts)* where *d* is the number of dimensions.
            A (1D) array of shape *(npts,)* is also allowed.
        """

        # This .T goofiness allows both the nD and the 1D case.
        return (np.dot(self.a, x).T + self.b).T

    # mypy limitation: "Decorated property not supported"
    @property           # type: ignore[misc]
    @memoize_method
    def jacobian(self):
        return la.det(self.a)

    # mypy limitation: "Decorated property not supported"
    @property           # type: ignore[misc]
    @memoize_method
    def inverse(self):
        """The inverse :class:`AffineMap` of *self*."""
        return AffineMap(la.inv(self.a), -la.solve(self.a, self.b))

# }}}


# {{{ simplex coordinate mapping

EQUILATERAL_TO_UNIT_MAP = {
        1: AffineMap([[1]], [0]),
        2: AffineMap([
            [1, -1/sqrt(3)],
            [0,  2/sqrt(3)]],
            [-1/3,   -1/3]),
        3: AffineMap([
            [1, -1/sqrt(3), -1/sqrt(6)],
            [0,  2/sqrt(3), -1/sqrt(6)],
            [0,         0,  sqrt(6)/2]],
            [-1/2,   -1/2,       -1/2])
        }


def equilateral_to_unit(equi):
    return EQUILATERAL_TO_UNIT_MAP[len(equi)](equi)


def unit_vertices(dim):
    from warnings import warn
    warn("unit_vertices is deprecated. "
            "Use modepy.unit_vertices_for_shape instead. "
            "unit_vertices will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    return shp.unit_vertices_for_shape(shp.Simplex(dim)).T


def barycentric_to_unit(bary):
    """
    :arg bary: shaped ``(dims+1, npoints)``
    """
    dims = len(bary)-1
    return np.dot(shp.unit_vertices_for_shape(shp.Simplex(dims)), bary)


def unit_to_barycentric(unit):
    """
    :arg unit: shaped ``(dims,npoints)``
    """

    last_bary = 0.5*(unit+1)
    first_bary = 1-np.sum(last_bary, axis=0)
    return np.vstack([first_bary, last_bary])


# /!\ do not reorder these, stuff (node generation) *will* break.
EQUILATERAL_VERTICES = {
        1: np.array([
            [-1],
            [1],
            ]),
        2: np.array([
            [-1, -1/sqrt(3)],
            [1,  -1/sqrt(3)],
            [0,   2/sqrt(3)],
            ]),
        3: np.array([
            [-1, -1/sqrt(3), -1/sqrt(6)],
            [1,  -1/sqrt(3), -1/sqrt(6)],
            [0,   2/sqrt(3), -1/sqrt(6)],
            [0,          0,   3/sqrt(6)],
            ])
        }


def barycentric_to_equilateral(bary):
    dims = len(bary)-1
    return np.dot(EQUILATERAL_VERTICES[dims].T, bary)

# }}}


# {{{ submeshes

def simplex_submesh(node_tuples):
    """Return a list of tuples of indices into the node list that
    generate a tesselation of the reference element.

    :arg node_tuples: A list of tuples *(i, j, ...)* of integers
        indicating node positions inside the unit element. The
        returned list references indices in this list.

        :func:`pytools.generate_nonnegative_integer_tuples_summing_to_at_most`
        may be used to generate *node_tuples*.
    """
    return shp.submesh_for_shape(shp.Simplex(len(node_tuples[0])), node_tuples)


submesh = MovedFunctionDeprecationWrapper(simplex_submesh)


def hypercube_submesh(node_tuples):
    """Return a list of tuples of indices into the node list that
    generate a tesselation of the reference element.

    :arg node_tuples: A list of tuples *(i, j, ...)* of integers
        indicating node positions inside the unit element. The
        returned list references indices in this list.

        :func:`pytools.generate_nonnegative_integer_tuples_below`
        may be used to generate *node_tuples*.

    See also :func:`simplex_submesh`.

    .. versionadded:: 2020.2
    """
    from warnings import warn
    warn("hypercube_submesh is deprecated. "
            "Use submesh_for_shape instead. "
            "hypercube_submesh will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    return shp.submesh_for_shape(shp.Hypercube(len(node_tuples[0])), node_tuples)

# }}}


# {{{ plotting helpers

def plot_element_values(n, nodes, values, resample_n=None,
        node_tuples=None, show_nodes=False):
    dims = len(nodes)

    orig_nodes = nodes
    orig_values = values

    if resample_n is not None:
        import modepy as mp
        basis = mp.simplex_onb(dims, n)
        fine_nodes = mp.equidistant_nodes(dims, resample_n)

        values = np.dot(mp.resampling_matrix(basis, fine_nodes, nodes), values)
        nodes = fine_nodes
        n = resample_n

    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
            as gnitstam

    if dims == 1:
        import matplotlib.pyplot as pt
        pt.plot(nodes[0], values)
        if show_nodes:
            pt.plot(orig_nodes[0], orig_values, "x")
        pt.show()
    elif dims == 2:
        import mayavi.mlab as mlab
        mlab.triangular_mesh(
                nodes[0], nodes[1], values, submesh(list(gnitstam(n, 2))))
        if show_nodes:
            mlab.points3d(orig_nodes[0], orig_nodes[1], orig_values,
                    scale_factor=0.05)
        mlab.show()
    else:
        raise RuntimeError("unsupported dimensionality %d" % dims)

# }}}


# {{{ lebesgue constant

def _evaluate_lebesgue_function(n, nodes, shape):
    huge_n = 30*n

    from modepy.spaces import space_for_shape
    from modepy.modes import basis_for_space
    from modepy.nodes import node_tuples_for_space
    space = space_for_shape(shape, n)
    huge_space = space_for_shape(shape, huge_n)

    basis = basis_for_space(space, shape)
    equi_node_tuples = node_tuples_for_space(huge_space)
    equi_nodes = (np.array(equi_node_tuples, dtype=np.float64)/huge_n*2 - 1).T
    assert equi_nodes.shape[0] == nodes.shape[0]

    from modepy.matrices import vandermonde
    vdm = vandermonde(basis.functions, nodes)

    eq_vdm = vandermonde(basis.functions, equi_nodes)
    eq_to_out = la.solve(vdm.T, eq_vdm.T).T

    lebesgue_worst = np.sum(np.abs(eq_to_out), axis=1)

    return lebesgue_worst, equi_node_tuples, equi_nodes


def estimate_lebesgue_constant(n, nodes, shape=None, visualize=False):
    """Estimate the
    `Lebesgue constant
    <https://en.wikipedia.org/wiki/Lebesgue_constant_(interpolation)>`_
    of the *nodes* at polynomial order *n*.

    :arg nodes: an array of shape *(dim, nnodes)* as returned by
        :func:`modepy.warp_and_blend_nodes`.
    :arg shape: a :class:`~modepy.shapes.Shape`.
    :arg visualize: visualize the function that gives rise to the
        returned Lebesgue constant. (2D only for now)
    :return: the Lebesgue constant, a scalar.

    .. versionadded:: 2013.2

    .. versionchanged:: 2020.2

        *domain* parameter was added with support for nodes on the unit
        hypercube (i.e. unit square in 2D and unit cube in 3D).

    .. versionchanged:: 2020.3

        Renamed *domain* to *shape*.
    """
    dim = len(nodes)
    if shape is None:
        from warnings import warn
        warn("Not passing shape is deprecated and will stop working "
                "in 2022.", DeprecationWarning, stacklevel=2)
        from modepy.shapes import Simplex
        shape = Simplex(dim)
    else:
        if shape.dim != dim:
            raise ValueError(f"expected {shape.dim}-dimensional nodes")

    lebesgue_worst, equi_node_tuples, equi_nodes = \
            _evaluate_lebesgue_function(n, nodes, shape)
    lebesgue_constant = np.max(lebesgue_worst)

    if not visualize:
        return lebesgue_constant

    if shape.dim == 1:
        import matplotlib.pyplot as plt
        plt.plot(equi_nodes[0], lebesgue_worst)
        plt.show()
    elif shape.dim == 2:
        print(f"Lebesgue constant: {lebesgue_constant}")
        triangles = shp.submesh_for_shape(shape, equi_node_tuples)

        try:
            import mayavi.mlab as mlab
            mlab.figure(bgcolor=(1, 1, 1))
            mlab.triangular_mesh(
                    equi_nodes[0], equi_nodes[1], lebesgue_worst / lebesgue_constant,
                    triangles)

            x, y = np.mgrid[-1:1:20j, -1:1:20j]
            mlab.mesh(x, y, 0*x,
                    representation="wireframe",
                    color=(0.4, 0.4, 0.4),
                    line_width=0.6)
            cb = mlab.colorbar()
            cb.label_text_property.color = (0, 0, 0)

            mlab.show()
        except ImportError:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.gca()
            ax.grid()
            ax.plot(nodes[0], nodes[1], "ko")
            # NOTE: might be tempted to use `plot_trisurf` here to get a plot
            # like mayavi, but that will be horrendously slow
            p = ax.tricontourf(
                    equi_nodes[0], equi_nodes[1], lebesgue_worst / lebesgue_constant,
                    triangles=triangles,
                    levels=16)
            fig.colorbar(p)
            ax.set_aspect("equal")
            plt.show()
    else:
        raise ValueError(f"visualization is not supported in {shape.dim}D")

    return lebesgue_constant

# }}}


# {{{ tensor product reshaping

@runtime_checkable
class Reshapeable(Protocol):
    shape: Tuple[int, ...]

    def reshape(
            self: "ReshapeableT", *newshape: Tuple[int, ...], order: str
            ) -> "ReshapeableT":
        ...


ReshapeableT = TypeVar("ReshapeableT", bound=Reshapeable)


def reshape_array_for_tensor_product_space(
        space: TensorProductSpace, ary: ReshapeableT, axis=-1) -> ReshapeableT:
    """Return a reshaped view of *ary* that exposes the tensor product nature
    of the space. Axis number *axis* of *ary* must index coefficients
    corresponding to a tensor-product-structured basis (e.g. modal or nodal
    coefficients).

    :arg ary: an array with last dimension having a length matching the
        :attr:`~modepy.FunctionSpace.space_dim` of *space*.
    :arg result: *ary* reshaped with axis number *axis* replaced by
        a tuple of dimensions matching the dimensions of the spaces
        making up the tensor product. Variation of the represented
        function along a given dimension will be represented by variation
        of array entries along the corresponding array axis.
    """
    if axis < 0:
        axis += len(ary.shape)
    if not (0 <= axis < len(ary.shape)):
        raise ValueError("invalid axis specified")
    if ary.shape[axis] != space.space_dim:
        raise ValueError(f"array's axis {axis} must have length "
                f"{space.space_dim}, found {ary.shape[axis]} instead")
    return ary.reshape(
            (ary.shape[:axis]
                + tuple([s.space_dim for s in space.bases])
                + ary.shape[axis+1:]),
            order="F")


def unreshape_array_for_tensor_product_space(
        space: TensorProductSpace, ary: ReshapeableT, axis=-1) -> ReshapeableT:
    """Undoes the effect of :func:`reshape_array_for_tensor_product_space`,
    given the same *space* and *axis*.
    """

    n_tp_axes = len(space.bases)
    naxes = len(ary.shape) - n_tp_axes + 1
    if axis < 0:
        axis += naxes
    if not (0 <= axis < naxes):
        raise ValueError("invalid axis specified")

    expected_space_dims = tuple([s.space_dim for s in space.bases])
    if ary.shape[axis:axis+n_tp_axes] != expected_space_dims:
        raise ValueError(f"array's axes {axis}:{axis+n_tp_axes} must have shape "
                f"{expected_space_dims}, "
                f"found {ary.shape[axis:axis+n_tp_axes]} instead")
    return ary.reshape(
            (ary.shape[:axis]
                + (space.space_dim,)
                + ary.shape[axis+n_tp_axes:]),
            order="F")

# }}}


# vim: foldmethod=marker
