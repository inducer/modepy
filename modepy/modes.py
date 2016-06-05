# - encoding: utf-8 -

from __future__ import division, absolute_import

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


from six.moves import range

from math import sqrt
import numpy as np
from modepy.tools import accept_scalar_or_vector


__doc__ = """:mod:`modepy.modes` provides orthonormal bases and their
derivatives on unit simplices.

Jacobi polynomials
------------------

.. currentmodule:: modepy

.. autofunction:: jacobi(alpha, beta, n, x)

.. autofunction:: grad_jacobi(alpha, beta, n, x)

Dimension-independent basis getters
-----------------------------------

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

.. autofunction:: simplex_onb

.. autofunction:: grad_simplex_onb

.. autofunction:: simplex_monomial_basis

.. autofunction:: grad_simplex_monomial_basis

Dimension-specific functions
----------------------------

.. currentmodule:: modepy.modes

.. autofunction:: pkdo_2d(order, rs)

.. autofunction:: grad_pkdo_2d(order, rs)

.. autofunction:: pkdo_3d(order, rst)

.. autofunction:: grad_pkdo_3d(order, rst)

Monomials
---------

.. autofunction:: monomial(order, rst)

.. autofunction:: grad_monomial(order, rst)
"""


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
    `Legendre polynomials <https://en.wikipedia.org/wiki/Legendre_polynomials>`_.
    """

    out_shape = x.shape
    x = x.ravel()

    from modepy.tools import gamma

    n = np.int32(n)
    Nx = len(x)
    if x.shape[0] > 1:
        x = x.T

    # Storage for recursive construction
    PL = np.zeros((Nx, n+1), np.float64)

    # Initial values P_0(x) and P_1(x)
    gamma0 = (2**(alpha+beta+1)
            / (alpha+beta+1.)
            * gamma(alpha+1) * gamma(beta+1) / gamma(alpha+beta+1))

    PL[:, 0] = 1.0/sqrt(gamma0)
    if n == 0:
        return PL[:, 0].reshape(out_shape)

    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
    PL[:, 1] = ((alpha+beta+2)*x/2 + (alpha-beta)/2)/sqrt(gamma1)
    if n == 1:
        return PL[:, 1].reshape(out_shape)

    # Repeat value in recurrence.
    aold = 2./(2.+alpha+beta)*sqrt((alpha+1.)*(beta+1.)/(alpha+beta+3.))

    # Forward recurrence using the symmetry of the recurrence.
    for i in range(1, n):
            h1 = 2.*i+alpha+beta

            foo = (i+1.)*(i+1.+alpha+beta)*(i+1.+alpha)*(i+1.+beta)/(h1+1.)/(h1+3.)
            anew = 2./(h1+2.)*sqrt(foo)

            bnew = -(alpha*alpha-beta*beta)/(h1*(h1+2.))
            PL[:, i+1] = (-aold*PL[:, i-1] + np.multiply(x-bnew, PL[:, i]))/anew
            aold = anew

    return PL[:, n].reshape(out_shape)


def grad_jacobi(alpha, beta, n, x):
    """Evaluate the derivative of :func:`jacobi`,
    with the same meanings and restrictions for all arguments.
    """
    if n == 0:
        return np.zeros_like(x)
    else:
        return sqrt(n*(n+alpha+beta+1)) \
                * jacobi(alpha+1, beta+1, n-1, x)

# }}}


# {{{ 2D PKDO

def _rstoab(r, s):
    """Transfer from (r, s) -> (a, b) coordinates in triangle.
    """

    a = np.empty_like(r)

    valid = (s != 1)
    a[valid] = 2*(1+r[valid])/(1-s[valid])-1
    a[~valid] = -1
    b = s
    return a, b


@accept_scalar_or_vector(arg_nr=2, expected_rank=2)
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


@accept_scalar_or_vector(arg_nr=2, expected_rank=2)
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

def _rsttoabc(r, s, t):
    Np = len(r)
    tol = 1e-10

    a = np.zeros(Np)
    b = np.zeros(Np)
    c = np.zeros(Np)

    for n in range(Np):
        if abs(s[n]+t[n]) > tol:
            a[n] = 2*(1+r[n])/(-s[n]-t[n])-1
        else:
            a[n] = -1

        if abs(t[n]-1.) > tol:
            b[n] = 2*(1+s[n])/(1-t[n])-1
        else:
            b[n] = -1

        c[n] = t[n]

    return a, b, c


@accept_scalar_or_vector(arg_nr=2, expected_rank=2)
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


@accept_scalar_or_vector(arg_nr=2, expected_rank=2)
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

@accept_scalar_or_vector(arg_nr=2, expected_rank=2)
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


@accept_scalar_or_vector(arg_nr=2, expected_rank=2)
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


# {{{ dimension-independent interface

def simplex_onb(dims, n):
    """Return a list of orthonormal basis functions in dimension *dims* of maximal
    total degree *n*.

    :returns: a class:`tuple` of functions, each of  which
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
    from functools import partial
    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
            as gnitstam

    if dims == 0:
        def zerod_basis(x):
            if len(x.shape) == 1:
                return 1
            else:
                return np.ones(x.shape[1])

        return (zerod_basis,)

    elif dims == 1:
        return tuple(partial(jacobi, 0, 0, i) for i in range(n+1))
    elif dims == 2:
        return tuple(partial(pkdo_2d, order) for order in gnitstam(n, dims))
    elif dims == 3:
        return tuple(partial(pkdo_3d, order) for order in gnitstam(n, dims))
    else:
        raise NotImplementedError("%d-dimensional bases" % dims)


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


def simplex_monomial_basis(dims, n):
    """Return a list of monomial basis functions in dimension *dims* of maximal
    total degree *n*.

    :returns: a class:`tuple` of functions, each of  which
        accepts arrays of shape *(dims, npts)*
        and return the function values as an array of size *npts*.
        'Scalar' evaluation, by passing just one vector of length *dims*,
        is also supported.

    .. versionadded:: 2016.1
    """

    from functools import partial
    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
            as gnitstam
    return tuple(partial(monomial, order) for order in gnitstam(n, dims))


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
    if dims <= 3:
        return simplex_onb(dims, n)
    else:
        return simplex_monomial_basis(dims, n)


# undocumented for now
def grad_simplex_best_available_basis(dims, n):
    if dims <= 3:
        return grad_simplex_onb(dims, n)
    else:
        return grad_simplex_monomial_basis(dims, n)

# }}}

# vim: foldmethod=marker
