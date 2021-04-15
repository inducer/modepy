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


from warnings import warn
from math import sqrt
from functools import singledispatch, partial

import numpy as np

from modepy.spaces import FunctionSpace, PN, QN
from modepy.shapes import Shape, Simplex, Hypercube

__doc__ = """This functionality provides sets of basis functions for the
reference elements in :mod:`modepy.shapes`.

.. currentmodule:: modepy

Basis Retrieval
---------------

.. autoexception:: BasisNotOrthonormal
.. autoclass:: Basis

.. autofunction:: basis_for_space
.. autofunction:: orthonormal_basis_for_space
.. autofunction:: monomial_basis_for_space

Jacobi polynomials
------------------

.. currentmodule:: modepy

.. autofunction:: jacobi(alpha, beta, n, x)
.. autofunction:: grad_jacobi(alpha, beta, n, x)

Conversion to Symbolic
----------------------
.. autofunction:: symbolicize_function

Tensor product adapter
----------------------

.. autoclass:: TensorProductBasis

PKDO basis functions
--------------------

.. currentmodule:: modepy.modes

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


.. autofunction:: pkdo_2d
.. autofunction:: grad_pkdo_2d
.. autofunction:: pkdo_3d
.. autofunction:: grad_pkdo_3d

Monomials
---------

.. autofunction:: monomial
.. autofunction:: grad_monomial

Redirections to Canonical Names
-------------------------------

.. currentmodule:: modepy.modes

.. class:: Basis

    See :class:`modepy.Basis`.
"""


# {{{ helpers for symbolic evaluation

def _cse(expr, prefix):
    from pymbolic.primitives import CommonSubexpression, Expression
    if isinstance(expr, Expression):
        return CommonSubexpression(expr, prefix)
    else:
        return expr

    return expr


def _where(op_a, comp, op_b, then, else_):
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

    fa = _cse(jacobi(0, 0, i, a), f"leg_{i}")
    dfa = _cse(grad_jacobi(0, 0, i, a), "dleg_{i}")
    gb = _cse(jacobi(2*i+1, 0, j, b), f"jac_{2*i+1}_{j}")
    dgb = _cse(grad_jacobi(2*i+1, 0, j, b), f"djac_{2*i+1}_{j}")

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

    fa = _cse(jacobi(0, 0, i, a), f"leg_{i}")
    dfa = _cse(grad_jacobi(0, 0, i, a), f"dleg_{i}")
    gb = _cse(jacobi(2*i+1, 0, j, b), f"jac_{2*i+1}")
    dgb = _cse(grad_jacobi(2*i+1, 0, j, b), f"djac_{2*i+1}")
    hc = _cse(jacobi(2*(i+j)+2, 0, k, c), f"jac_{2*(i+j)+2}")
    dhc = _cse(grad_jacobi(2*(i+j)+2, 0, k, c), f"djac_{2*(i+j)+2}")

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


# {{{ DEPRECATED dimension-independent interface for simplices

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
    warn("simplex_onb_with_mode_ids is deprecated. "
            "Use orthonormal_basis_for_space instead. "
            "This function will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    if dims == 1:
        mode_ids = tuple(range(n+1))
        return mode_ids, tuple(partial(jacobi, 0, 0, i) for i in mode_ids)
    else:
        b = _SimplexONB(dims, n)
        return b.mode_ids, b.functions


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
    warn("simplex_onb is deprecated. "
            "Use orthonormal_basis_for_space instead. "
            "This function will go away in 2022.",
            DeprecationWarning, stacklevel=2)

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
    warn("grad_simplex_onb is deprecated. "
            "Use orthonormal_basis_for_space instead. "
            "This function will go away in 2022.",
            DeprecationWarning, stacklevel=2)

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
    warn("simplex_monomial_basis_with_mode_ids is deprecated. "
            "Use monomial_basis_for_space instead. "
            "This function will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    from modepy.nodes import node_tuples_for_space
    mode_ids = node_tuples_for_space(PN(dims, n))

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

    warn("grad_simplex_monomial_basis_with_mode_ids is deprecated. "
            "Use monomial_basis_for_space instead. "
            "This function will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
            as gnitstam
    return tuple(partial(grad_monomial, order) for order in gnitstam(n, dims))


def simplex_best_available_basis(dims, n):
    warn("simplex_best_available_basis is deprecated. "
            "Use basis_for_space instead. "
            "This function will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    return basis_for_space(PN(dims, n), Simplex(dims)).functions


def grad_simplex_best_available_basis(dims, n):
    warn("grad_simplex_best_available_basis is deprecated. "
            "Use basis_for_space instead. "
            "This function will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    return basis_for_space(PN(dims, n), Simplex(dims)).gradients

# }}}


# {{{ tensor product basis helpers

class _TensorProductBasisFunction:
    # multi_index is here just for debugging.

    def __init__(self, multi_index, per_dim_functions):
        self.multi_index = multi_index
        self.per_dim_functions = per_dim_functions

    def __call__(self, x):
        result = 1
        for iaxis, per_dim_function in enumerate(self.per_dim_functions):
            result = result * per_dim_function(x[iaxis])

        return result


class _TensorProductGradientBasisFunction:
    # multi_index is here just for debugging.

    def __init__(self, multi_index, per_dim_derivatives):
        self.multi_index = multi_index
        self.per_dim_derivatives = tuple(per_dim_derivatives)

    def __call__(self, x):
        result = [1] * len(self.per_dim_derivatives)
        for ider, per_dim_functions in enumerate(self.per_dim_derivatives):
            for iaxis, per_dim_function in enumerate(per_dim_functions):
                result[ider] *= per_dim_function(x[iaxis])

        return tuple(result)

# }}}


# {{{ DEPRECATED dimension-independent basis getters

def tensor_product_basis(dims, basis_1d):
    """Adapt any iterable *basis_1d* of 1D basis functions into a *dims*-dimensional
    tensor product basis.

    :returns: a tuple of callables representing a *dims*-dimensional basis

    .. versionadded:: 2017.1
    """
    warn("tensor_product_basis is deprecated. "
            "Use TensorProductBasis instead. "
            "This function will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    from modepy.nodes import node_tuples_for_space
    mode_ids = node_tuples_for_space(QN(dims, len(basis_1d) - 1))

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
    warn("grad_tensor_product_basis is deprecated. "
            "Use TensorProductBasis instead. "
            "This function will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    from pytools import wandering_element
    from modepy.nodes import node_tuples_for_space
    mode_ids = node_tuples_for_space(QN(dims, len(basis_1d) - 1))

    func = (basis_1d, grad_basis_1d)
    return tuple(
            _TensorProductGradientBasisFunction(order, [
                [func[i][k] for i, k in zip(iderivative, order)]
                for iderivative in wandering_element(dims)
                ])
            for order in mode_ids)


def legendre_tensor_product_basis(dims, order):
    warn("legendre_tensor_product_basis is deprecated. "
            "Use orthonormal_basis_for_space instead. "
            "This function will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    basis = [partial(jacobi, 0, 0, n) for n in range(order + 1)]
    return tensor_product_basis(dims, basis)


def grad_legendre_tensor_product_basis(dims, order):
    warn("grad_legendre_tensor_product_basis is deprecated. "
            "Use orthonormal_basis_for_space instead. "
            "This function will go away in 2022.",
            DeprecationWarning, stacklevel=2)

    basis = [partial(jacobi, 0, 0, n) for n in range(order + 1)]
    grad_basis = [partial(grad_jacobi, 0, 0, n) for n in range(order + 1)]
    return grad_tensor_product_basis(dims, basis, grad_basis)

# }}}


# {{{ conversion to symbolic

def symbolicize_function(f, dim, ref_coord_var_name="r"):
    """For a function *f* (basis or gradient) returned by one of the functions in
    this module, return a :mod:`pymbolic` expression representing the
    same function.

    :arg dim: the number of dimensions of the reference element on which
        *basis* is defined.

    .. versionadded:: 2020.2
    """
    import pymbolic.primitives as p
    r_sym = p.make_sym_vector(ref_coord_var_name, dim)

    result = f(r_sym)

    if dim == 1:
        # Work around inconsistent 1D stupidity. Grrrr!
        # (We fed it an object array, and it gave one back, i.e. it treated its
        # argument as a scalar instead of indexing into it. That tends to
        # happen for 1D functions. Because we're aiming for future consistency
        # across 1D/nD, we'll first try to feed *every* basis object arrays and
        # only recover if it does the wrong/inconsistent thing.)
        if isinstance(result, np.ndarray) and result.dtype.char == "O":
            r_sym = p.Variable("r")[0]
            return f(r_sym)
        else:
            return result
    else:
        return result

# }}}


# {{{ basis interface

class BasisNotOrthonormal(Exception):
    pass


class Basis:
    """
    .. automethod:: orthonormality_weight
    .. autoattribute:: mode_ids
    .. autoattribute:: functions
    .. autoattribute:: gradients
    """

    def orthonormality_weight(self):
        raise NotImplementedError

    @property
    def mode_ids(self):
        """Return a tuple of of mode (basis function) identifiers, one for
        each basis function. Mode identifiers should generally be viewed as opaque.
        They are hashable. For some bases, they are tuples of length matching
        the number of dimensions and indicating the order along each reference
        axis.
        """
        raise NotImplementedError

    @property
    def functions(self):
        """Return a tuple of (callable) basis functions of length matching
        :attr:`mode_ids`.  Each function takes a vector of (r,s,t) reference
        coordinates as input.
        """
        raise NotImplementedError

    @property
    def gradients(self):
        """Return a tuple of basis functions of length matching :attr:`mode_ids`.
        Each function takes a vector of (r,s,t) reference coordinates as input.
        """
        raise NotImplementedError

# }}}


# {{{ space-based basis retrieval

@singledispatch
def basis_for_space(space: FunctionSpace, shape: Shape) -> Basis:
    raise NotImplementedError(type(space).__name__)


@singledispatch
def orthonormal_basis_for_space(space: FunctionSpace, shape: Shape) -> Basis:
    raise NotImplementedError(type(space).__name__)


@singledispatch
def monomial_basis_for_space(space: FunctionSpace, shape: Shape) -> Basis:
    raise NotImplementedError(type(space).__name__)

# }}}


def zerod_basis(x):
    assert len(x) == 0
    x_sub = np.ones(x.shape[1:], x.dtype)
    return 1 + x_sub


# {{{ PN bases

def _pkdo_1d(order, r):
    i, = order
    r0, = r
    return jacobi(0, 0, i, r0)


def _grad_pkdo_1d(order, r):
    i, = order
    r0, = r
    return (grad_jacobi(0, 0, i, r0),)


class _SimplexBasis(Basis):
    def __init__(self, dim, order):
        self._dim = dim
        self._order = order

        assert isinstance(dim, int)
        assert isinstance(order, int)

    @property
    def mode_ids(self):
        from pytools import \
                generate_nonnegative_integer_tuples_summing_to_at_most as gnitsam
        return tuple(gnitsam(self._order, self._dim))


class _SimplexONB(_SimplexBasis):
    is_orthonormal = True

    def orthonormality_weight(self):
        return 1

    @property
    def functions(self):
        if self._dim == 0:
            return (zerod_basis,)
        elif self._dim == 1:
            return tuple(partial(_pkdo_1d, mid) for mid in self.mode_ids)
        elif self._dim == 2:
            return tuple(partial(pkdo_2d, mid) for mid in self.mode_ids)
        elif self._dim == 3:
            return tuple(partial(pkdo_3d, mid) for mid in self.mode_ids)
        else:
            raise NotImplementedError(f"basis in {self._dim} dimensions")

    @property
    def gradients(self):
        if self._dim == 1:
            return tuple(partial(_grad_pkdo_1d, mid) for mid in self.mode_ids)
        elif self._dim == 2:
            return tuple(partial(grad_pkdo_2d, mid) for mid in self.mode_ids)
        elif self._dim == 3:
            return tuple(partial(grad_pkdo_3d, mid) for mid in self.mode_ids)
        else:
            raise NotImplementedError(f"gradient in {self._dim} dimensions")


class _SimplexMonomialBasis(_SimplexBasis):
    def orthonormality_weight(self):
        raise BasisNotOrthonormal

    @property
    def functions(self):
        return tuple(partial(monomial, mid) for mid in self.mode_ids)

    @property
    def gradients(self):
        return tuple(partial(grad_monomial, mid) for mid in self.mode_ids)


@basis_for_space.register(PN)
def _(space: PN, shape: Simplex):
    if not isinstance(shape, Simplex):
        raise NotImplementedError((type(space).__name__, type(shape).__name))

    if space.spatial_dim <= 3:
        return _SimplexONB(space.spatial_dim, space.order)
    else:
        return _SimplexMonomialBasis(space.spatial_dim, space.order)


@orthonormal_basis_for_space.register(PN)
def _(space: PN, shape: Simplex):
    if not isinstance(shape, Simplex):
        raise NotImplementedError((type(space).__name__, type(shape).__name))

    return _SimplexONB(space.spatial_dim, space.order)


@monomial_basis_for_space.register(PN)
def _(space: PN, shape: Simplex):
    if not isinstance(shape, Simplex):
        raise NotImplementedError((type(space).__name__, type(shape).__name))

    return _SimplexMonomialBasis(space.spatial_dim, space.order)

# }}}


# {{{ QN bases

class TensorProductBasis(Basis):
    """Adapts multiple one-dimensional bases into a tensor product basis.

    .. automethod:: __init__
    """

    def __init__(self, bases_1d, grad_bases_1d, orth_weight):
        """
        :arg bases_1d: a sequence (one entry per axis/dimension)
            of sequences (representing the basis) of 1D functions
            representing the approximation basis.
        :arg grad_bases_1d: a sequence (one entry per axis/dimension)
            representing the derivatives of *bases_1d*.
        """
        self._bases_1d = bases_1d
        self._grad_bases_1d = grad_bases_1d
        self._orth_weight = orth_weight

        if len(bases_1d) != len(grad_bases_1d):
            raise ValueError("bases_1d and grad_bases_1d must have the same length")

        for i, (b, gb) in enumerate(zip(bases_1d, grad_bases_1d)):
            if len(b) != len(gb):
                raise ValueError(
                        f"bases_1d[{i}] and grad_bases_1d[{i}] "
                        "must have the same length")

    def orthonormality_weight(self):
        if self._orth_weight is None:
            raise BasisNotOrthonormal
        else:
            return self._orth_weight

    @property
    def _dim(self):
        return len(self._bases_1d)

    @property
    def mode_ids(self):
        from pytools import generate_nonnegative_integer_tuples_below as gnitb
        return tuple(gnitb([len(b) for b in self._bases_1d]))

    @property
    def functions(self):
        return tuple(
                _TensorProductBasisFunction(mid,
                    [self._bases_1d[iaxis][mid_i]
                        for iaxis, mid_i in enumerate(mid)])
                for mid in self.mode_ids)

    @property
    def gradients(self):
        from pytools import wandering_element
        func = (self._bases_1d, self._grad_bases_1d)
        return tuple(
                _TensorProductGradientBasisFunction(mid, [
                    [func[is_deriv][iaxis][mid_i]
                        for iaxis, (is_deriv, mid_i) in
                        enumerate(zip(iderivative, mid))]
                    for iderivative in wandering_element(self._dim)
                    ])
                for mid in self.mode_ids)


@orthonormal_basis_for_space.register(QN)
def _(space: QN, shape: Hypercube):
    if not isinstance(shape, Hypercube):
        raise NotImplementedError((type(space).__name__, type(shape).__name))

    order = space.order
    dim = space.spatial_dim
    return TensorProductBasis(
            [[partial(jacobi, 0, 0, n) for n in range(order + 1)]] * dim,
            [[partial(grad_jacobi, 0, 0, n) for n in range(order + 1)]] * dim,
            orth_weight=1)


@basis_for_space.register(QN)
def _(space: QN, shape: Hypercube):
    if not isinstance(shape, Hypercube):
        raise NotImplementedError((type(space).__name__, type(shape).__name))

    return orthonormal_basis_for_space(space, shape)


def _monomial_1d(order, r):
    return r**order


def _grad_monomial_1d(order, r):
    if order == 0:
        return 0*r
    else:
        return order*r**(order-1)


@monomial_basis_for_space.register(QN)
def _(space: QN, shape: Hypercube):
    if not isinstance(shape, Hypercube):
        raise NotImplementedError((type(space).__name__, type(shape).__name))

    order = space.order
    dim = space.spatial_dim
    return TensorProductBasis(
            [[partial(_monomial_1d, n) for n in range(order + 1)]] * dim,
            [[partial(_grad_monomial_1d, n) for n in range(order + 1)]] * dim,
            orth_weight=None)

# }}}

# vim: foldmethod=marker
