from __future__ import division

__copyright__ = "Copyright (C) 2009, 2010, 2013 Andreas Kloeckner, Tim Warburton, Jan Hesthaven, Xueyu Zhu"

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




from math import sqrt
import numpy as np




__doc__ = """:mod:`modepy.modes` provides orthonormal bases and their
derivatives on unit simplices.
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

    from modepy.tools import gamma

    n = np.int32(n)
    Nx = len(x)
    if x.shape[0]>1:
        x = x.T

    # Storage for recursive construction
    PL = np.zeros((Nx, n+1), np.float64)

    # Initial values P_0(x) and P_1(x)
    gamma0 = (2**(alpha+beta+1)
            / (alpha+beta+1.)
            * gamma(alpha+1) * gamma(beta+1) / gamma(alpha+beta+1))

    PL[:,0] = 1.0/sqrt(gamma0)
    if n == 0:
        return PL[:,0]

    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
    PL[:,1] = ((alpha+beta+2)*x/2 + (alpha-beta)/2)/sqrt(gamma1)
    if n==1:
        return PL[:,1]

    # Repeat value in recurrence.
    aold = 2./(2.+alpha+beta)*sqrt((alpha+1.)*(beta+1.)/(alpha+beta+3.))

    # Forward recurrence using the symmetry of the recurrence.
    for i in xrange(1, n):
            h1 = 2.*i+alpha+beta

            foo = (i+1.)*(i+1.+alpha+beta)*(i+1.+alpha)*(i+1.+beta)/(h1+1.)/(h1+3.)
            anew = 2./(h1+2.)*sqrt(foo)

            bnew = -(alpha*alpha-beta*beta)/(h1*(h1+2.))
            PL[:, i+1] = ( -aold*PL[:, i-1] + np.multiply(x-bnew, PL[:, i]) )/anew
            aold = anew

    return PL[:, n]




def grad_jacobi(alpha, beta, n, x):
    """Evaluate the derivative of :func:`jacobi`,
    with the same meanings and restrictions for all arguments.
    """
    if n == 0:
        return np.zeros(len(x))
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

    fa  = jacobi     (0, 0, i, a)
    dfa = grad_jacobi(0, 0, i, a)
    gb  = jacobi     (2*i+1, 0, j, b)
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
        if abs(s[n]+t[n])>tol:
            a[n] = 2*(1+r[n])/(-s[n]-t[n])-1
        else:
            a[n] = -1

        if abs(t[n]-1.)>tol:
            b[n] = 2*(1+s[n])/(1-t[n])-1
        else:
            b[n] = -1

        c[n] = t[n]

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




def grad_pkdo_3d(a, b, c, id, jd, kd):
    """Return the derivatives of the modal basis (id, jd, kd) on the
    3D simplex at (a, b, c).

    See the following publications:

    * |proriol-ref|
    * |koornwinder-ref|
    * |dubiner-ref|
    """

    fa  = JacobiP(a, 0, 0, id).reshape(len(a),1)
    dfa = GradJacobiP(a, 0, 0, id)
    gb  = JacobiP(b, 2*id+1,0, jd).reshape(len(b),1)
    dgb = GradJacobiP(b, 2*id+1,0, jd)
    hc  = JacobiP(c, 2*(id+jd)+2,0, kd).reshape(len(c),1)
    dhc = GradJacobiP(c, 2*(id+jd)+2,0, kd)

    # r-derivative
    # d/dr = da/dr d/da + db/dr d/db + dc/dr d/dx
    dmodedr = dfa*gb*hc
    if(id>0):
        dmodedr = dmodedr*((0.5*(1-b))**(id-1))
    if(id+jd>0):
        dmodedr = dmodedr*((0.5*(1-c))**(id+jd-1))

    # s-derivative
    dmodeds = 0.5*(1+a)*dmodedr
    tmp = dgb*((0.5*(1-b))**id)
    if(id>0):
        tmp = tmp+(-0.5*id)*(gb*(0.5*(1-b))**(id-1))

    if(id+jd>0):
        tmp = tmp*((0.5*(1-c))**(id+jd-1))

    tmp = fa*tmp*hc
    dmodeds = dmodeds + tmp

    # t-derivative
    dmodedt = 0.5*(1+a)*dmodedr+0.5*(1+b)*tmp
    tmp = dhc*((0.5*(1-c))**(id+jd))
    if(id+jd>0):
        tmp = tmp-0.5*(id+jd)*(hc*((0.5*(1-c))**(id+jd-1)));

    tmp = fa*(gb*tmp)
    tmp = tmp*((0.5*(1-b))**id);
    dmodedt = dmodedt+tmp;

    # Normalize
    dmodedr = 2**(2*id+jd+1.5)*dmodedr
    dmodeds = 2**(2*id+jd+1.5)*dmodeds
    dmodedt = 2**(2*id+jd+1.5)*dmodedt

    return dmodedr[:,0], dmodeds[:,0], dmodedt[:,0]

# }}}

# {{{ dimension-independent interface

def get_simplex_onb(dims, p):
    """
    See the following publications:

    * |proriol-ref|
    * |koornwinder-ref|
    * |dubiner-ref|
    """
    from functools import partial
    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
            as gnitstam

    if dims == 1:
        return [partial(jacobi, 0, 0, i) for i in xrange(p+1)]
    elif dims == 2:
        return [partial(pkdo_2d, order) for order in gnitstam(p, dims)]
    elif dims == 3:
        return [partial(pkdo_3d, order) for order in gnitstam(p, dims)]
    else:
        raise NotImplementedError("%d-dimensional bases" % dims)

def get_grad_simplex_onb(dims, p):
    """
    See the following publications:

    * |proriol-ref|
    * |koornwinder-ref|
    * |dubiner-ref|
    """
    from functools import partial
    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
            as gnitstam

    if dims == 1:
        return [partial(grad_jacobi, 0, 0, i) for i in xrange(p+1)]
    elif dims == 2:
        return [partial(grad_pkdo_2d, order) for order in gnitstam(p, dims)]
    else:
        raise NotImplementedError("%d-dimensional bases" % dims)

# }}}

# vim: foldmethod=marker
