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

from functools import reduce

import numpy as np
import numpy.linalg as la
from math import sqrt
from pytools import memoize_method, MovedFunctionDeprecationWrapper


try:
    # Python 2.7 and newer
    from math import gamma
except ImportError:
    _have_gamma = False
else:
    _have_gamma = True


if not _have_gamma:
    try:
        from scipy.special import gamma  # noqa
    except ImportError:
        pass
    else:
        _have_gamma = True


if not _have_gamma:
    def gamma(z):  # noqa
        from warnings import warn
        warn("Using makeshift gamma function that only works for integers. "
                "No better one was found.")

        if z != int(z):
            raise RuntimeError("makeshift gamma function doesn't work "
                    "for non-integers")

        g = 1
        for i in range(1, int(z)):
            g = g*i

        return g


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
        from pytools import factorial
        from operator import mul

        return (self.factor * 2**len(self.exponents)
                * reduce(mul, (factorial(alpha) for alpha in self.exponents))
                / factorial(len(self.exponents)+sum(self.exponents)))

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

    @property
    @memoize_method
    def jacobian(self):
        return la.det(self.a)

    @property
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
    result = np.empty((dim+1, dim), np.float64)
    result.fill(-1)

    for i in range(dim):
        result[i+1, i] = 1

    return result


# this should go away
UNIT_VERTICES = {
        0: unit_vertices(0),
        1: unit_vertices(1),
        2: unit_vertices(2),
        3: unit_vertices(3),
        }


def barycentric_to_unit(bary):
    """
    :arg bary: shaped ``(dims+1,npoints)``
    """
    dims = len(bary)-1
    return np.dot(unit_vertices(dims).T, bary)


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


# {{{ hypercube coordinate mapping

def hypercube_unit_vertices(dims):
    from modepy.nodes import tensor_product_nodes
    return tensor_product_nodes(dims, np.array([-1.0, 1.0])).T

# }}}


def pick_random_simplex_unit_coordinate(rng, dims):
    offset = 0.05
    base = -1 + offset
    remaining = 2 - dims*offset
    r = np.zeros(dims, np.float64)
    for j in range(dims):
        rn = rng.uniform(0, remaining)
        r[j] = base + rn
        remaining -= rn
    return r


def pick_random_hypercube_unit_coordinate(rng, dims):
    return np.array([rng.uniform(-1.0, 1.0) for _ in range(dims)])

# {{{ accept_scalar_or_vector decorator

class accept_scalar_or_vector:  # noqa
    def __init__(self, arg_nr, expected_rank):
        """
        :arg arg_nr: The argument number which may be a scalar or a vector,
            one-based.
        """
        self.arg_nr = arg_nr - 1
        self.expected_rank = expected_rank

    def __call__(self, f):

        def wrapper(*args, **kwargs):
            controlling_arg = args[self.arg_nr]
            try:
                shape = controlling_arg.shape
            except AttributeError:
                has_shape = False
            else:
                has_shape = True

            if not has_shape:
                if not self.expected_rank == 1:
                    raise ValueError("cannot pass a scalar to %s" % f)

                controlling_arg = np.array([controlling_arg])
                new_args = args[:self.arg_nr] \
                        + (controlling_arg,) + args[self.arg_nr+1:]
                result = f(*new_args, **kwargs)

                if isinstance(result, tuple):
                    return tuple(r[0] for r in result)
                else:
                    return result[0]

            if len(shape) == self.expected_rank:
                return f(*args, **kwargs)
            elif len(shape) < self.expected_rank:
                controlling_arg = controlling_arg[..., np.newaxis]

                new_args = args[:self.arg_nr] \
                        + (controlling_arg,) + args[self.arg_nr+1:]
                result = f(*new_args, **kwargs)

                if isinstance(result, tuple):
                    return tuple(r[..., 0] for r in result)
                else:
                    return result[..., 0]
            else:
                raise ValueError("argument rank is too large: got %d, expected %d"
                        % (len(shape), self.expected_rank))

        from functools import wraps
        try:
            wrapper = wraps(f)(wrapper)
        except AttributeError:
            pass

        return wrapper

# }}}


# {{{ submeshes, plotting helpers

def simplex_submesh(node_tuples):
    """Return a list of tuples of indices into the node list that
    generate a tesselation of the reference element.

    :arg node_tuples: A list of tuples *(i, j, ...)* of integers
        indicating node positions inside the unit element. The
        returned list references indices in this list.

        :func:`pytools.generate_nonnegative_integer_tuples_summing_to_at_most`
        may be used to generate *node_tuples*.
    """
    from pytools import single_valued, add_tuples
    dims = single_valued(len(nt) for nt in node_tuples)

    node_dict = {
            ituple: idx
            for idx, ituple in enumerate(node_tuples)}

    if dims == 1:
        result = []

        def try_add_line(d1, d2):
            try:
                result.append((
                    node_dict[add_tuples(current, d1)],
                    node_dict[add_tuples(current, d2)],
                    ))
            except KeyError:
                pass

        for current in node_tuples:
            try_add_line((0,), (1,),)

        return result
    elif dims == 2:
        # {{{ triangle sub-mesh
        result = []

        def try_add_tri(d1, d2, d3):
            try:
                result.append((
                    node_dict[add_tuples(current, d1)],
                    node_dict[add_tuples(current, d2)],
                    node_dict[add_tuples(current, d3)],
                    ))
            except KeyError:
                pass

        for current in node_tuples:
            # this is a tesselation of a square into two triangles.
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
            try:
                result.append((
                    node_dict[add_tuples(current, d1)],
                    node_dict[add_tuples(current, d2)],
                    node_dict[add_tuples(current, d3)],
                    node_dict[add_tuples(current, d4)],
                    ))
            except KeyError:
                pass

        result = []
        for current in node_tuples:
            # this is a tesselation of a cube into six tets.
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
        raise NotImplementedError("%d-dimensional sub-meshes" % dims)


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

    from pytools import single_valued, add_tuples
    dims = single_valued(len(nt) for nt in node_tuples)

    node_dict = {
            ituple: idx
            for idx, ituple in enumerate(node_tuples)}

    from pytools import generate_nonnegative_integer_tuples_below as gnitb

    result = []
    for current in node_tuples:
        try:
            result.append(tuple(
                    node_dict[add_tuples(current, offset)]
                    for offset in gnitb(2, dims)))

        except KeyError:
            pass

    return result


@accept_scalar_or_vector(2, 2)
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

def _evaluate_lebesgue_function(n, nodes, domain):
    dims = len(nodes)
    huge_n = 30*n

    if domain == "simplex":
        from modepy.modes import simplex_onb as domain_basis_onb
        from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as generate_node_tuples)
    elif domain == "hypercube":
        from modepy.modes import (
                legendre_tensor_product_basis as domain_basis_onb)
        from pytools import (
                generate_nonnegative_integer_tuples_below
                as generate_node_tuples)
    else:
        raise ValueError(f"unknown domain: '{domain}'")

    basis = domain_basis_onb(dims, n)
    equi_node_tuples = list(generate_node_tuples(huge_n, dims))
    equi_nodes = (np.array(equi_node_tuples, dtype=np.float64)/huge_n*2 - 1).T

    from modepy.matrices import vandermonde
    vdm = vandermonde(basis, nodes)

    eq_vdm = vandermonde(basis, equi_nodes)
    eq_to_out = la.solve(vdm.T, eq_vdm.T).T

    lebesgue_worst = np.sum(np.abs(eq_to_out), axis=1)

    return lebesgue_worst, equi_node_tuples, equi_nodes


def estimate_lebesgue_constant(n, nodes, domain=None, visualize=False):
    """Estimate the
    `Lebesgue constant
    <https://en.wikipedia.org/wiki/Lebesgue_constant_(interpolation)>`_
    of the *nodes* at polynomial order *n*.

    :arg nodes: an array of shape *(dims, nnodes)* as returned by
        :func:`modepy.warp_and_blend_nodes`.
    :arg domain: represents the domain of the reference element and can be
        either ``"simplex"`` or ``"hypercube"``.
    :arg visualize: visualize the function that gives rise to the
        returned Lebesgue constant. (2D only for now)
    :return: the Lebesgue constant, a scalar.

    .. versionadded:: 2013.2

    .. versionchanged:: 2020.2

        *domain* parameter was added with support for nodes on the unit
        hypercube (i.e. unit square in 2D and unit cube in 3D).
    """
    if domain is None:
        domain = "simplex"

    dims = len(nodes)
    lebesgue_worst, equi_node_tuples, equi_nodes = \
            _evaluate_lebesgue_function(n, nodes, domain)
    lebesgue_constant = np.max(lebesgue_worst)

    if not visualize:
        return lebesgue_constant

    if dims == 2:
        print(f"Lebesgue constant: {lebesgue_constant}")

        if domain == "simplex":
            triangles = simplex_submesh(equi_node_tuples)
        elif domain == "hypercube":
            triangles = hypercube_submesh(equi_node_tuples)
        else:
            triangles = None

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
        raise ValueError(f"visualization is not supported in {dims}D")

    return lebesgue_constant

# }}}


# vim: foldmethod=marker
