from __future__ import division

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




import numpy as np
import numpy.linalg as la
from math import sqrt
from pytools import memoize_method, memoize




try:
    # Python 2.7 and newer
    from math import gamma
except ImportError:
    _have_gamma = False
else:
    _have_gamma = True

if not _have_gamma:
    try:
        from scipy.special import gamma
    except ImportError:
        pass
    else:
        _have_gamma = True

if not _have_gamma:
    raise RuntimeError("no usable implementation of the gamma function found")







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
        return self.factor* \
                reduce(mul, (x[i]**expn
                    for i, expn in enumerate(self.exponents)))

    def simplex_integral(self):
        """Integral over the unit simplex."""
        from pytools import factorial
        from operator import mul

        return (self.factor*2**len(self.exponents)*
            reduce(mul, (factorial(alpha) for alpha in self.exponents))
            /
            factorial(len(self.exponents)+sum(self.exponents)))

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
        self.a = a
        self.b = b

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




@memoize
def get_equilateral_to_unit(dims):
    if dims == 1:
        return AffineMap(
                np.array([[1]], dtype=np.float64),
                np.array([0], dtype=np.float64))
    elif dims == 2:
        return AffineMap(
                np.array([[1, -1 / sqrt(3)], [0, 2 / sqrt(3)]]),
                    np.array([-1/3, -1/3]))
    elif dims == 3:
        return AffineMap(
                np.array([
                    [1, -1/sqrt(3), -1/sqrt(6)],
                    [0,  2/sqrt(3), -1/sqrt(6)],
                    [0,         0,  sqrt(6)/2]
                    ]),
                    np.array([-1/2, -1/2, -1/2]))
    else:
        raise NotImplementedError("%d-dimensional map" % dims)

# }}}

# {{{ submeshes

def get_submesh(node_tuples):
    """Return a list of tuples of indices into the node list that
    generate a tesselation of the reference element.

    :arg node_tuples: A list of tuples *(i, j, ...)* of integers
        indicating node positions inside the unit element. The
        returned list references indices in this list.
    """
    from pytools import single_valued, add_tuples
    dims = single_valued(len(nt) for nt in node_tuples)

    node_dict = dict(
            (ituple, idx)
            for idx, ituple in enumerate(node_tuples))

    if dims == 2:
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

# }}}

class accept_scalar_or_vector:
    def __init__(self, arg_nr, expected_rank):
        """
        :arg arg_nr: The argument number which may be a scalar or a vector,
            one-based.
        """
        self.arg_nr = arg_nr - 1
        self.expected_rank = expected_rank

    def __call__(self, f):
        from functools import wraps

        @wraps(f)
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
                new_args = args[:self.arg_nr] + (controlling_arg,) + args[self.arg_nr+1:]
                result = f(*new_args, **kwargs)

                if isinstance(result, tuple):
                    return tuple(r[0] for r in result)
                else:
                    return result[0]

            if len(shape) == self.expected_rank:
                return f(*args, **kwargs)
            else:
                controlling_arg = controlling_arg[..., np.newaxis]

                new_args = args[:self.arg_nr] + (controlling_arg,) + args[self.arg_nr+1:]
                result = f(*new_args, **kwargs)

                if isinstance(result, tuple):
                    return tuple(r[..., 0] for r in result)
                else:
                    return result[..., 0]

        return wrapper

# vim: foldmethod=marker
