__copyright__ = ("Copyright (C) 2009, 2010, 2013 "
"Andreas Kloeckner, Tim Warburton, Jan Hesthaven, Xueyu Zhu")

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


import sys
from math import sqrt
import numpy as np

from modepy.shapes import Simplex, Hypercube
from modepy.shapes import get_basis, get_grad_basis, get_basis_with_mode_ids


__doc__ = """:mod:`modepy.modes` provides orthonormal bases and their
derivatives on unit simplices.

Jacobi polynomials
------------------

.. currentmodule:: modepy

.. autofunction:: jacobi(alpha, beta, n, x)

.. autofunction:: grad_jacobi(alpha, beta, n, x)

Dimension-independent basis getters for simplices
-------------------------------------------------

.. |proriol-ref| replace::
    Proriol, Joseph. "Sur une famille de polynomes á deux variables orthogonaux
    dans un triangle." CR Acad. Sci. Paris 245 (1957): 2459-2461.

.. |koornwinder-ref| replace::
    Koornwinder, T. "Two-variable analogues of the classical orthogonal polynomials."
    Theory and Applications of Special Functions. 1975, pp. 435-495.

.. |dubiner-ref| replace::
    Dubiner, Moshe. "Spectral Methods on Triangles and Other Domains." Journal of
    Scientific Computing 6, no. 4 (December 1, 1991): 345–390.
    http://dx.doi.org/10.1007/BF01060030

.. autofunction:: simplex_onb_with_mode_ids
.. autofunction:: simplex_onb
.. autofunction:: grad_simplex_onb
.. autofunction:: simplex_monomial_basis_with_mode_ids
.. autofunction:: simplex_monomial_basis
.. autofunction:: grad_simplex_monomial_basis

Dimension-independent basis getters for tensor-product bases
------------------------------------------------------------

.. autofunction:: tensor_product_basis
.. autofunction:: grad_tensor_product_basis

Dimension-specific functions
----------------------------

.. currentmodule:: modepy.modes

.. autofunction:: pkdo_2d
.. autofunction:: grad_pkdo_2d
.. autofunction:: pkdo_3d
.. autofunction:: grad_pkdo_3d

Monomials
---------

.. autofunction:: monomial
.. autofunction:: grad_monomial

Symbolic Basis Functions
------------------------
.. autofunction:: symbolicize_basis
"""


# {{{ shape basis functions

# {{{ simplex

@get_basis.register(Simplex)
def _(shape: Simplex, order: int):
    if shape.dims <= 3:
        return simplex_onb(shape.dims, order)
    else:
        return simplex_monomial_basis(shape.dims, order)


@get_grad_basis.register(Simplex)
def _(shape: Simplex, order: int):
    if shape.dims <= 3:
        return grad_simplex_onb(shape.dims, order)
    else:
        return grad_simplex_monomial_basis(shape.dims, order)


@get_basis_with_mode_ids.register(Simplex)
def _(shape: Simplex, order: int):
    if shape.dims <= 3:
        return simplex_onb_with_mode_ids(shape.dims, order)
    else:
        return simplex_monomial_basis_with_mode_ids(shape.dims, order)

# }}}


# {{{ hypercube

@get_basis.register(Hypercube)
def _(shape: Hypercube, order: int):
    return legendre_tensor_product_basis(shape.dims, order)


@get_grad_basis.register(Hypercube)
def _(shape: Hypercube, order: int):
    return grad_legendre_tensor_product_basis(shape.dims, order)


@get_basis_with_mode_ids.register(Hypercube)
def _(shape: Hypercube, order: int):
    from modepy.shapes import get_node_tuples
    mode_ids = get_node_tuples(shape, order)
    return mode_ids, get_basis(shape, order)

# }}}

# }}}


# {{{ helpers for symbolic evaluation

def _cse(expr, prefix):
    if "pymbolic" in sys.modules:
        from pymbolic.primitives import CommonSubexpression, Expression
        if isinstance(expr, Expression):
            return CommonSubexpression(expr, prefix)
        else:
            return expr

    return expr


def _where(op_a, comp, op_b, then, else_):
    if "pymbolic" in sys.modules:
        from pymbolic.primitives import If, Comparison, Expression
        if isinstance(op_a, Expression) or isinstance(op_b, Expression):
            return If(Comparison(op_a, comp, op_b), then, else_)

    import operator
    comp_op = getattr(operator, comp)

    if isinstance(op_a, np.ndarray) or isinstance(op_b, np.ndarray):
        return np.where(comp_op(op_a, op_b), then, else_)

    return then if comp_op(op_a, op_b) else else_

# }}}


# {{{ jacobi polynomials

def jacobi(alpha, beta, n, x):
    r"""Evaluate `Jacobi polynomials
    <https://en.wikipedia.org/wiki/Jacobi_polynomials>`_ of type :math:`(\alpha,
    \beta)` with :math:`\alpha, \beta > -1` and :math:`\alpha+\beta \ne -1` at a
    vector of points *x* for order *n*. The points *x* must lie on the interval
    :math:`[-1,1]`.

    :return: a vector of :math:`P^{(\alpha, \beta)}_n` evaluated at all *x*.

    The polynomials are normalized to be orthonormal with respect to the
    Jacobi weight :math:`(1-x)^\alpha(1+x)^\beta`.

    Observe that choosing :math:`\alpha=\beta=0` will yield the
    `Legendre polynomials <https://en.wikipedia.org/wiki/Legendre_polynomials>`__.
    """

    from modepy.tools import gamma

    # Initial values P_0(x) and P_1(x)
    # NOTE: general formula gets a divide by 0 in the `alpha + beta == -1` case,
    # so we replace it by a nicer limit using
    #   lim 2**(x + 1) / (x + 1) / gamma(x + 1) = 1 as x -> -1
    if abs(alpha + beta + 1) < 1.0e-14:
        gamma0 = gamma(alpha+1) * gamma(beta+1)
    else:
        gamma0 = (2**(alpha+beta+1) / (alpha+beta+1.)
                * gamma(alpha+1) * gamma(beta+1) / gamma(alpha+beta+1))

    # Storage for recursive construction
    pl = [1.0/sqrt(gamma0) + 0*x]

    if n == 0:
        return pl[0]

    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0

    pl.append(_cse(
        ((alpha+beta+2)*x/2 + (alpha-beta)/2)/sqrt(gamma1),
        prefix="jac_p1"))
    if n == 1:
        return pl[1]

    # Repeat value in recurrence.
    aold = 2./(2.+alpha+beta)*sqrt((alpha+1.)*(beta+1.)/(alpha+beta+3.))

    # Forward recurrence using the symmetry of the recurrence.
    for i in range(1, n):
        h1 = 2.*i+alpha+beta

        foo = (i+1.)*(i+1.+alpha+beta)*(i+1.+alpha)*(i+1.+beta)/(h1+1.)/(h1+3.)
        anew = 2./(h1+2.)*sqrt(foo)

        bnew = -(alpha*alpha-beta*beta)/(h1*(h1+2.))
        pl.append(_cse(
            (-aold*pl[i-1] + np.multiply(x-bnew, pl[i]))/anew,
            prefix=f"jac_p{i+1}"))
        aold = anew

    return pl[n]


def grad_jacobi(alpha, beta, n, x):
    """Evaluate the derivative of :func:`jacobi`,
    with the same meanings and restrictions for all arguments.
    """
    if n == 0:
        return 0*x
    else:
        return sqrt(n*(n+alpha+beta+1)) * jacobi(alpha+1, beta+1, n-1, x)

# }}}


# {{{ 2D PKDO

def _rstoab(r, s, tol=1e-12):
    """Transfer from (r, s) -> (a, b) coordinates in triangle.
    """

    # We may divide by zero below (or close to it), but we won't use the
    # results because of the conditional. Silence the resulting numpy warnings.
    with np.errstate(all="ignore"):
        a = _where(abs(s-1), "ge", tol, 2*(1+r)/(1-s)-1, -1)
    b = s
    return a, b


def pkdo_2d(order, rs):
    """Evaluate a 2D orthonormal (with weight 1) polynomial on the unit simplex.

    :arg order: A tuple *(i, j)* representing the order of the polynomial.
    :arg rs: ``rs[0], rs[1]`` are arrays of :math:`(r,s)` coordinates.
        (See :ref:`tri-coords`)
    :return: a vector of values of the same length as the *rs* arrays.

    See the following publications:

    * |proriol-ref|
    * |koornwinder-ref|
    * |dubiner-ref|
    """

    a, b = _rstoab(*rs)
    i, j = order

    h1 = jacobi(0, 0, i, a)
    h2 = jacobi(2*i+1, 0, j, b)
    return sqrt(2)*h1*h2*(1-b)**i


def grad_pkdo_2d(order, rs):
    """Evaluate the derivatives of :func:`pkdo_2d`.

    :arg order: A tuple *(i, j)* representing the order of the polynomial.
    :arg rs: ``rs[0], rs[1]`` are arrays of :math:`(r,s)` coordinates.
        (See :ref:`tri-coords`)
    :return: a tuple of vectors *(dphi_dr, dphi_ds)*, each of the same length
        as the *rs* arrays.

    See the following publications:

    * |proriol-ref|
    * |koornwinder-ref|
    * |dubiner-ref|
    """

    a, b = _rstoab(*rs)
    i, j = order

    fa = jacobi(0, 0, i, a)
    dfa = grad_jacobi(0, 0, i, a)
    gb = jacobi(2*i+1, 0, j, b)
    dgb = grad_jacobi(2*i+1, 0, j, b)

    # r-derivative
    # d/dr
    #  = da/dr d/da + db/dr d/db
    #  = (2/(1-s)) d/da
    #  = (2/(1-b)) d/da
    dmodedr = dfa*gb
    if i:
        dmodedr = dmodedr*((0.5*(1-b))**(i-1))

    # s-derivative
    # d/ds = ((1+a)/2)/((1-b)/2) d/da + d/db
    dmodeds = dfa*(gb*(0.5*(1+a)))
    if i:
        dmodeds = dmodeds*((0.5*(1-b))**(i-1))
    tmp = dgb*((0.5*(1-b))**i)
    if i:
        tmp = tmp-0.5*i*gb*((0.5*(1-b))**(i-1))
    dmodeds = dmodeds+fa*tmp

    # Normalize
    dmodedr = 2**(i+0.5)*dmodedr
    dmodeds = 2**(i+0.5)*dmodeds

    return dmodedr, dmodeds

# }}}


# {{{ 3D PKDO

def _rsttoabc(r, s, t, tol=1e-10):
    # We may divide by zero below (or close to it), but we won't use the
    # results because of the conditional. Silence the resulting numpy warnings.
    with np.errstate(all="ignore"):
        a = _where(abs(s+t), "gt", tol, 2*(1+r)/(-s-t)-1, -1)
        b = _where(abs(t-1.), "gt", tol, 2*(1+s)/(1-t)-1, -1)
        c = t

    return a, b, c


def pkdo_3d(order, rst):
    """Evaluate a 2D orthonormal (with weight 1) polynomial on the unit simplex.

    :arg order: A tuple *(i, j, k)* representing the order of the polynomial.
    :arg rs: ``rst[0], rst[1], rst[2]`` are arrays of :math:`(r,s,t)` coordinates.
        (See :ref:`tet-coords`)
    :return: a vector of values of the same length as the *rst* arrays.

    See the following publications:

    * |proriol-ref|
    * |koornwinder-ref|
    * |dubiner-ref|
    """

    a, b, c = _rsttoabc(*rst)
    i, j, k = order

    h1 = jacobi(0, 0, i, a)
    h2 = jacobi(2*i+1, 0, j, b)
    h3 = jacobi(2*(i+j)+2, 0, k, c)

    return 2*sqrt(2)*h1*h2*((1-b)**i)*h3*((1-c)**(i+j))


def grad_pkdo_3d(order, rst):
    """Evaluate the derivatives of :func:`pkdo_3d`.

    :arg order: A tuple *(i, j, k)* representing the order of the polynomial.
    :arg rs: ``rs[0], rs[1], rs[2]`` are arrays of :math:`(r,s,t)` coordinates.
        (See :ref:`tet-coords`)
    :return: a tuple of vectors *(dphi_dr, dphi_ds, dphi_dt)*, each of the same
        length as the *rst* arrays.

    See the following publications:

    * |proriol-ref|
    * |koornwinder-ref|
    * |dubiner-ref|
    """

    a, b, c = _rsttoabc(*rst)
    i, j, k = order

    fa = jacobi(0, 0, i, a)
    dfa = grad_jacobi(0, 0, i, a)
    gb = jacobi(2*i+1, 0, j, b)
    dgb = grad_jacobi(2*i+1, 0, j, b)
    hc = jacobi(2*(i+j)+2, 0, k, c)
    dhc = grad_jacobi(2*(i+j)+2, 0, k, c)

    # r-derivative
    # d/dr = da/dr d/da + db/dr d/db + dc/dr d/dx
    dmodedr = dfa*gb*hc
    if i:
        dmodedr = dmodedr*((0.5*(1-b))**(i-1))
    if i+j:
        dmodedr = dmodedr*((0.5*(1-c))**(i+j-1))

    # s-derivative
    dmodeds = 0.5*(1+a)*dmodedr
    tmp = dgb*((0.5*(1-b))**i)
    if i:
        tmp = tmp+(-0.5*i)*(gb*(0.5*(1-b))**(i-1))

    if i+j:
        tmp = tmp*((0.5*(1-c))**(i+j-1))

    tmp = fa*tmp*hc
    dmodeds = dmodeds + tmp

    # t-derivative
    dmodedt = 0.5*(1+a)*dmodedr+0.5*(1+b)*tmp
    tmp = dhc*((0.5*(1-c))**(i+j))
    if i+j:
        tmp = tmp-0.5*(i+j)*(hc*((0.5*(1-c))**(i+j-1)))

    tmp = fa*(gb*tmp)
    tmp = tmp*((0.5*(1-b))**i)
    dmodedt = dmodedt+tmp

    # Normalize
    dmodedr = 2**(2*i+j+1.5)*dmodedr
    dmodeds = 2**(2*i+j+1.5)*dmodeds
    dmodedt = 2**(2*i+j+1.5)*dmodedt

    return dmodedr, dmodeds, dmodedt

# }}}


# {{{ monomials

def monomial(order, rst):
    """Evaluate the monomial of order *order* at the points *rst*.

    :arg order: A tuple *(i, j,...)* representing the order of the polynomial.
    :arg rst: ``rst[0], rst[1]`` are arrays of :math:`(r,s,...)` coordinates.
        (See :ref:`tri-coords`)
    """
    dim = len(order)
    assert dim == rst.shape[0]

    from pytools import product
    return product(rst[i] ** order[i] for i in range(dim))


def grad_monomial(order, rst):
    """Evaluate the derivative of the monomial of order *order* at the points *rst*.

    :arg order: A tuple *(i, j,...)* representing the order of the polynomial.
    :arg rst: ``rst[0], rst[1]`` are arrays of :math:`(r,s,...)` coordinates.
        (See :ref:`tri-coords`)
    :return: a tuple of vectors *(dphi_dr, dphi_ds, dphi_dt)*, each of the same
        length as the *rst* arrays.

    .. versionadded:: 2016.1
    """
    dim = len(order)
    assert dim == rst.shape[0]

    def diff_monomial(r, o):
        if o == 0:
            return 0*r
        elif o == 1:
            return 1+0*r
        else:
            return o * r**(o-1)

    from pytools import product
    return tuple(
            product(
                (
                    diff_monomial(rst[i], order[i])
                    if j == i else
                    rst[i] ** order[i])
                for i in range(dim)
                )
            for j in range(dim))

# }}}


# {{{ dimension-independent interface for simplices

def zerod_basis(x):
    if len(x.shape) == 1:
        return 1
    else:
        return np.ones(x.shape[1])


def simplex_onb_with_mode_ids(dims, n):
    """Return a list of orthonormal basis functions in dimension *dims* of maximal
    total degree *n*.

    :returns: a tuple ``(mode_ids, basis)``, where *basis* is a class:`tuple`
        of functions, each of  which accepts arrays of shape *(dims, npts)* and
        return the function values as an array of size *npts*.  'Scalar'
        evaluation, by passing just one vector of length *dims*, is also supported.
        *mode_ids* is a tuple of the same length as *basis*, where each entry
        is a tuple of integers describing the order of the mode along the coordinate
        axes.

    See the following publications:

    * |proriol-ref|
    * |koornwinder-ref|
    * |dubiner-ref|

    .. versionadded:: 2018.1
    """
    from modepy.shapes import get_node_tuples
    shape = Simplex(dims)

    from functools import partial
    if dims == 0:
        mode_ids = get_node_tuples(shape, n)
        return mode_ids, (zerod_basis,)
    elif dims == 1:
        # FIXME: should also use get_node_tuples
        mode_ids = tuple(range(n+1))
        return mode_ids, tuple(partial(jacobi, 0, 0, i) for i in mode_ids)
    elif dims == 2:
        mode_ids = get_node_tuples(shape, n)
        return mode_ids, tuple(partial(pkdo_2d, order) for order in mode_ids)
    elif dims == 3:
        mode_ids = get_node_tuples(shape, n)
        return mode_ids, tuple(partial(pkdo_3d, order) for order in mode_ids)
    else:
        raise NotImplementedError("%d-dimensional bases" % dims)


def simplex_onb(dims, n):
    """Return a list of orthonormal basis functions in dimension *dims* of maximal
    total degree *n*.

    :returns: a :class:`tuple` of functions, each of  which
        accepts arrays of shape *(dims, npts)*
        and return the function values as an array of size *npts*.
        'Scalar' evaluation, by passing just one vector of length *dims*,
        is also supported.

    See the following publications:

    * |proriol-ref|
    * |koornwinder-ref|
    * |dubiner-ref|

    .. versionchanged:: 2013.2

        Made return value a tuple, to make bases hashable.
    """
    mode_ids, basis = simplex_onb_with_mode_ids(dims, n)
    return basis


def grad_simplex_onb(dims, n):
    """Return the gradients of the functions returned by :func:`simplex_onb`.

    :returns: a :class:`tuple` of functions, each of  which
        accepts arrays of shape *(dims, npts)*
        and returns a :class:`tuple` of length *dims* containing
        the derivatives along each axis as an array of size *npts*.
        'Scalar' evaluation, by passing just one vector of length *dims*,
        is also supported.

    See the following publications:

    * |proriol-ref|
    * |koornwinder-ref|
    * |dubiner-ref|

    .. versionchanged:: 2013.2

        Made return value a tuple, to make bases hashable.
    """
    from functools import partial
    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
            as gnitstam

    if dims == 1:
        return tuple(partial(grad_jacobi, 0, 0, i) for i in range(n+1))
    elif dims == 2:
        return tuple(partial(grad_pkdo_2d, order) for order in gnitstam(n, dims))
    elif dims == 3:
        return tuple(partial(grad_pkdo_3d, order) for order in gnitstam(n, dims))
    else:
        raise NotImplementedError("%d-dimensional bases" % dims)


def simplex_monomial_basis_with_mode_ids(dims, n):
    """Return a list of monomial basis functions in dimension *dims* of maximal
    total degree *n*.

    :returns: a tuple ``(mode_ids, basis)``, where *basis* is a class:`tuple`
        of functions, each of  which accepts arrays of shape *(dims, npts)* and
        return the function values as an array of size *npts*.  'Scalar'
        evaluation, by passing just one vector of length *dims*, is also supported.
        *mode_ids* is a tuple of the same length as *basis*, where each entry
        is a tuple of integers describing the order of the mode along the coordinate
        axes.

    .. versionadded:: 2018.1
    """
    from modepy.shapes import get_node_tuples
    mode_ids = get_node_tuples(Simplex(dims), n)

    from functools import partial
    return mode_ids, tuple(partial(monomial, order) for order in mode_ids)


def simplex_monomial_basis(dims, n):
    """Return a list of monomial basis functions in dimension *dims* of maximal
    total degree *n*.

    :returns: a :class:`tuple` of functions, each of  which
        accepts arrays of shape *(dims, npts)*
        and return the function values as an array of size *npts*.
        'Scalar' evaluation, by passing just one vector of length *dims*,
        is also supported.

    .. versionadded:: 2016.1
    """
    mode_ids, basis = simplex_monomial_basis_with_mode_ids(dims, n)
    return basis


def grad_simplex_monomial_basis(dims, n):
    """Return the gradients of the functions returned by
    :func:`simplex_monomial_basis`.

    :returns: a :class:`tuple` of functions, each of  which
        accepts arrays of shape *(dims, npts)*
        and returns a :class:`tuple` of length *dims* containing
        the derivatives along each axis as an array of size *npts*.
        'Scalar' evaluation, by passing just one vector of length *dims*,
        is also supported.

    .. versionadded:: 2016.1
    """

    from functools import partial
    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
            as gnitstam
    return tuple(partial(grad_monomial, order) for order in gnitstam(n, dims))


# undocumented for now
def simplex_best_available_basis(dims, n):
    return get_basis(Simplex(dims), n)


# undocumented for now
def grad_simplex_best_available_basis(dims, n):
    return get_grad_basis(Simplex(dims), n)

# }}}


# {{{ tensor product basis

class _TensorProductBasisFunction:
    def __init__(self, multi_index, per_dim_functions):
        self.multi_index = multi_index
        self.per_dim_functions = per_dim_functions

    def __call__(self, x):
        result = 1
        for iaxis, per_dim_function in enumerate(self.per_dim_functions):
            result = result * per_dim_function(x[iaxis])

        return result


class _TensorProductGradientBasisFunction:
    def __init__(self, multi_index, per_dim_derivatives):
        self.multi_index = multi_index
        self.per_dim_derivatives = tuple(per_dim_derivatives)

    def __call__(self, x):
        result = [1] * len(self.per_dim_derivatives)
        for ider, per_dim_functions in enumerate(self.per_dim_derivatives):
            for iaxis, per_dim_function in enumerate(per_dim_functions):
                result[ider] *= per_dim_function(x[iaxis])

        return tuple(result)


def tensor_product_basis(dims, basis_1d):
    """Adapt any iterable *basis_1d* of 1D basis functions into a *dims*-dimensional
    tensor product basis.

    :returns: a tuple of callables representing a *dims*-dimensional basis

    .. versionadded:: 2017.1
    """
    from modepy.shapes import Hypercube, get_node_tuples
    mode_ids = get_node_tuples(Hypercube(dims), len(basis_1d))

    return tuple(
            _TensorProductBasisFunction(order, [basis_1d[i] for i in order])
            for order in mode_ids)


def grad_tensor_product_basis(dims, basis_1d, grad_basis_1d):
    """Provides derivatives for each of the basis functions generated by
    :func:`tensor_product_basis`.

    :returns: a :class:`tuple` of callables, where each one returns a
        *dims*-dimensional :class:`tuple`, one for each derivative.

    .. versionadded:: 2020.2
    """
    from pytools import wandering_element
    from modepy.shapes import Hypercube, get_node_tuples
    mode_ids = get_node_tuples(Hypercube(dims), len(basis_1d))

    func = (basis_1d, grad_basis_1d)
    return tuple(
            _TensorProductGradientBasisFunction(order, [
                [func[i][k] for i, k in zip(iderivative, order)]
                for iderivative in wandering_element(dims)
                ])
            for order in mode_ids)


def legendre_tensor_product_basis(dims, order):
    from functools import partial
    basis = [partial(jacobi, 0, 0, n) for n in range(order + 1)]
    return tensor_product_basis(dims, basis)


def grad_legendre_tensor_product_basis(dims, order):
    from functools import partial
    basis = [partial(jacobi, 0, 0, n) for n in range(order + 1)]
    grad_basis = [partial(grad_jacobi, 0, 0, n) for n in range(order + 1)]
    return grad_tensor_product_basis(dims, basis, grad_basis)

# }}}


# {{{ symbolic basis functions

def symbolicize_basis(basis, dims, ref_coord_var_name="r"):
    """For a basis or a gradient of a basis returned by one of the functions in
    this module, return a list of :mod:`pymbolic` expressions representing the
    same basis.

    :arg dims: the number of dimensions of the reference element on which
        *basis* is defined.

    .. versionadded:: 2020.2
    """
    import pymbolic.primitives as p
    r_sym = p.make_sym_vector(ref_coord_var_name, dims)

    result = [func(r_sym) for func in basis]

    if dims == 1:
        # Work around inconsistent 1D stupidity. Grrrr!
        # (We fed it an object array, and it gave one back, i.e. it treated its
        # argument as a scalar instead of indexing into it. That tends to
        # happen for 1D functions. Because we're aiming for future consistency
        # across 1D/nD, we'll first try to feed *every* basis object arrays and
        # only recover if it does the wrong/inconsistent thing.)
        if any(isinstance(sym_func, np.ndarray) and sym_func.dtype.char == "O"
                for sym_func in result):
            r_sym = p.Variable("r")[0]
            return [func(r_sym) for func in basis]
        else:
            return result
    else:
        return result

# }}}

# vim: foldmethod=marker
