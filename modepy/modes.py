from __future__ import annotations


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

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial, singledispatch
from typing import (
    TYPE_CHECKING,
    TypeVar,
    cast,
)

import numpy as np

from modepy.shapes import Shape, Simplex, TensorProductShape
from modepy.spaces import PN, FunctionSpace, TensorProductSpace


if TYPE_CHECKING:
    import pymbolic.primitives

RealValueT = TypeVar("RealValueT",
                     np.ndarray, "pymbolic.primitives.Expression", float)
""":class:`~typing.TypeVar` for basis function inputs and outputs."""

__doc__ = """This functionality provides sets of basis functions for the
reference elements in :mod:`modepy.shapes`.

.. autoclass:: RealValueT

.. currentmodule:: modepy

Basis Retrieval
---------------

.. autoexception:: BasisNotOrthonormal
.. autoclass:: Basis
    :members:

.. autofunction:: basis_for_space
.. autofunction:: orthonormal_basis_for_space
.. autofunction:: monomial_basis_for_space

Jacobi polynomials
------------------

.. currentmodule:: modepy

.. autofunction:: jacobi
.. autofunction:: grad_jacobi
.. autofunction:: scaled_jacobi

Conversion to Symbolic
----------------------

.. autofunction:: symbolicize_function

Tensor product adapter
----------------------

.. autoclass:: TensorProductBasis
    :members:
    :show-inheritance:

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
"""


# {{{ helpers for symbolic evaluation

def _cse(expr, prefix):
    from pymbolic.primitives import CommonSubexpression, ExpressionNode
    if isinstance(expr, ExpressionNode):
        return CommonSubexpression(expr, prefix)
    else:
        return expr


def _where(op_a, comp, op_b, then, else_):
    from pymbolic.primitives import Comparison, ExpressionNode, If
    if isinstance(op_a, ExpressionNode) or isinstance(op_b, ExpressionNode):
        return If(
              Comparison(op_a, Comparison.name_to_operator[comp], op_b),
              then, else_)

    import operator
    comp_op = getattr(operator, comp)

    if isinstance(op_a, np.ndarray) or isinstance(op_b, np.ndarray):
        return np.where(comp_op(op_a, op_b), then, else_)

    return then if comp_op(op_a, op_b) else else_

# }}}


# {{{ jacobi polynomials

def jacobi(alpha: float, beta: float, n: int, x: RealValueT) -> RealValueT:
    r"""Evaluate `Jacobi polynomials
    <https://en.wikipedia.org/wiki/Jacobi_polynomials>`_ of type
    :math:`(\alpha, \beta)`, with :math:`\alpha, \beta > -1`, and order *n*
    at a vector of points *x*. The points *x* must lie on the interval
    :math:`[-1, 1]`.

    The polynomials are normalized to be orthonormal with respect to the
    Jacobi weight :math:`(1 - x)^\alpha (1 + x)^\beta`.

    Observe that choosing :math:`\alpha = \beta = 0` will yield the
    `Legendre polynomials <https://en.wikipedia.org/wiki/Legendre_polynomials>`__.

    :returns: a vector of :math:`P^{(\alpha, \beta)}_n` evaluated at all *x*.
    """

    from math import gamma

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
    pl = [1.0/math.sqrt(gamma0) + 0*x]

    if n == 0:
        return pl[0]

    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0

    pl.append(_cse(
        ((alpha+beta+2)*x/2 + (alpha-beta)/2)/math.sqrt(gamma1),
        prefix="jac_p1"))
    if n == 1:
        return pl[1]

    # Repeat value in recurrence.
    aold = 2./(2.+alpha+beta)*math.sqrt((alpha+1.)*(beta+1.)/(alpha+beta+3.))

    # Forward recurrence using the symmetry of the recurrence.
    for i in range(1, n):
        h1 = 2.*i+alpha+beta

        foo = (i+1.)*(i+1.+alpha+beta)*(i+1.+alpha)*(i+1.+beta)/(h1+1.)/(h1+3.)
        anew = 2./(h1+2.)*math.sqrt(foo)

        bnew = -(alpha*alpha-beta*beta)/(h1*(h1+2.))
        pl.append(_cse(
            (-aold*pl[i-1] + (x-bnew) * pl[i])/anew,
            prefix=f"jac_p{i+1}"))
        aold = anew

    return pl[n]


def grad_jacobi(alpha: float, beta: float, n: int, x: RealValueT) -> RealValueT:
    """Evaluate the derivative of :func:`jacobi`, with the same meanings and
    restrictions for all arguments.
    """
    if n == 0:
        return 0*x
    else:
        return math.sqrt(n*(n+alpha+beta+1)) * jacobi(alpha+1, beta+1, n-1, x)


def binom(x, y):
    # FIXME This may overflow quickly.
    # mpmath has clever 'gammaprod' pole handling in case that's ever needed:
    # https://github.com/mpmath/mpmath/blob/75a2ed37c4f2c576a9d01d360ee4c94ead57c7ff/mpmath/functions/factorials.py#L61-L65
    from math import gamma
    return gamma(x + 1) / (gamma(y + 1) * gamma(x - y + 1))


def scaled_jacobi(alpha: float, beta: float, n: int, x: RealValueT) -> RealValueT:
    r"""
    Same as :func:`jacobi`, but scaled so that

    .. math::

        P_n^{(\alpha, \beta)}(1) = \binom{ n+\alpha \\ n},

    and

    .. math::

        P_n^{(\alpha, \beta)}(-1) = (-1)^n \binom{ n+\beta \\ n}.

    This is the more conventional definition of the Jacobi polynomials.
    """
    from math import factorial, gamma, sqrt

    eps = 100 * np.finfo((np.float32(0) + np.asarray(x)).dtype).eps
    if n == 0 and abs(alpha + beta + 1) < eps:
        # maxima claims limit(1/(x*gamma(x)),x, 0) == 1
        scaling = (
            # 2**(alpha + beta + 1) \approx 1
            gamma(n + alpha + 1)
            * gamma(n + beta + 1)
            / factorial(n)
        )
    else:
        # source:
        # https://en.wikipedia.org/w/index.php?title=Jacobi_polynomials&oldid=1219118998
        scaling = (
            2**(alpha + beta + 1)
            / (2*n + alpha + beta + 1)
            * gamma(n + alpha + 1)
            * gamma(n + beta + 1)
            / gamma(n + alpha + beta + 1)
            / factorial(n)
        )

    return sqrt(scaling) * jacobi(alpha, beta, n, x)

# }}}


# {{{ 2D PKDO

def _rstoab(
        r: RealValueT, s: RealValueT,
        tol: float = 1.0e-12) -> tuple[RealValueT, RealValueT]:
    """Transfer from (r, s) -> (a, b) coordinates in triangle."""

    # We may divide by zero below (or close to it), but we won't use the
    # results because of the conditional. Silence the resulting numpy warnings.
    with np.errstate(all="ignore"):
        a = _where(abs(s-1), "ge", tol, 2*(1+r)/(1-s)-1, -1)
    b = s
    return a, b


def pkdo_2d(order: tuple[int, int], rs: np.ndarray) -> np.ndarray:
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
    return math.sqrt(2)*h1*h2*(1-b)**i


def grad_pkdo_2d(
        order: tuple[int, int],
        rs: np.ndarray) -> tuple[RealValueT, RealValueT]:
    """Evaluate the derivatives of :func:`pkdo_2d`.

    :arg order: A tuple *(i, j)* representing the order of the polynomial.
    :arg rs: ``rs[0], rs[1]`` are arrays of :math:`(r, s)` coordinates.
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
    dfa = _cse(grad_jacobi(0, 0, i, a), f"dleg_{i}")
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

def _rsttoabc(
        r: RealValueT, s: RealValueT, t: RealValueT,
        tol: float = 1.0e-10) -> tuple[RealValueT, RealValueT, RealValueT]:
    # We may divide by zero below (or close to it), but we won't use the
    # results because of the conditional. Silence the resulting numpy warnings.
    with np.errstate(all="ignore"):
        a = _where(abs(s+t), "gt", tol, 2*(1+r)/(-s-t)-1, -1)
        b = _where(abs(t-1.), "gt", tol, 2*(1+s)/(1-t)-1, -1)
        c = t

    return a, b, c


def pkdo_3d(order: tuple[int, int, int], rst: np.ndarray) -> np.ndarray:
    """Evaluate a 2D orthonormal (with weight 1) polynomial on the unit simplex.

    :arg order: A tuple *(i, j, k)* representing the order of the polynomial.
    :arg rs: ``rst[0], rst[1], rst[2]`` are arrays of :math:`(r, s, t)`
        coordinates. (See :ref:`tet-coords`)
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

    return 2*math.sqrt(2)*h1*h2*((1-b)**i)*h3*((1-c)**(i+j))


def grad_pkdo_3d(
        order: tuple[int, int, int],
        rst: np.ndarray) -> tuple[RealValueT, RealValueT, RealValueT]:
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

def monomial(order: tuple[int, ...], rst: np.ndarray) -> np.ndarray:
    """Evaluate the monomial of order *order* at the points *rst*.

    :arg order: A tuple *(i, j,...)* representing the order of the polynomial.
    :arg rst: ``rst[0], rst[1]`` are arrays of :math:`(r, s, ...)` coordinates.
        (See :ref:`tri-coords`)
    """
    dim = len(order)
    assert dim == rst.shape[0]

    from pytools import product
    return product(rst[i] ** order[i] for i in range(dim))


def _diff_monomial_1d(r: np.ndarray, o: int) -> np.ndarray:
    if o == 0:
        return 0*r
    elif o == 1:
        return 1+0*r
    else:
        return o * r**(o-1)


def grad_monomial(order: tuple[int, ...], rst: np.ndarray) -> tuple[RealValueT, ...]:
    """Evaluate the derivative of the monomial of order *order* at the points *rst*.

    :arg order: A tuple *(i, j,...)* representing the order of the polynomial.
    :arg rst: ``rst[0], rst[1]`` are arrays of :math:`(r, s, ...)` coordinates.
        (See :ref:`tri-coords`)
    :return: a tuple of vectors *(dphi_dr, dphi_ds, dphi_dt, ....)*, each
        of the same length as the *rst* arrays.

    .. versionadded:: 2016.1
    """
    dim = len(order)
    assert dim == rst.shape[0]

    from pytools import product
    return tuple(
            product(
                (
                    _diff_monomial_1d(rst[i], order[i])
                    if j == i else
                    rst[i] ** order[i])
                for i in range(dim)
                )
            for j in range(dim))

# }}}


# {{{ tensor product basis helpers

@dataclass(frozen=True)
class _TensorProductBasisFunction:
    r"""
    .. autoattribute:: multi_index
    .. autoattribute:: functions
    .. autoattribute:: dims_per_function
    """

    multi_index: tuple[Hashable, ...]
    """
    A :class:`tuple` used to identify each function in :attr:`functions`
    that is mainly meant for debugging and not used internally.
    """

    functions: tuple[Callable[[np.ndarray], np.ndarray], ...]
    r"""A :class:`tuple` of callables that can be evaluated on the tensor
    product space :math:`\mathbb{R}^{d_1} \times \cdots \times \mathbb{R}^{d_n}`,
    i.e. one function :math:`f_i` for each :math:`\mathbb{R}^{d_i}` component

    .. math::

        f(x_1, \dots, x_d) =
                f_1(x_1, \dots, x_{d_1}) \times \cdots \times
                f_n(x_{d_{n - 1}}, \dots, x_{d_n})
    """

    dims_per_function: tuple[int, ...] = field(kw_only=True)
    r"""
    A :class:`tuple` containing the dimensions :math:`(d_1, \dots, d_n)`.
    """

    def __post_init__(self) -> None:
        assert len(self.dims_per_function) == len(self.functions)

    @property
    def ndim(self) -> int:
        return sum(self.dims_per_function)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[0] == self.ndim

        n = 0

        if x.shape[1:]:
            # This ensures the result has the right shape even
            # if we're in zero dimensions.
            # The dtype gymnastics are intended as a way to coerce the
            # dtype to be floating point without changing the accuracy.
            result = np.ones(x.shape[1:], dtype=(x+np.float32(1)).dtype)
        else:
            # Likely we're evaluating symbolically.
            result = 1  # type: ignore[assignment]

        for d, function in zip(self.dims_per_function, self.functions, strict=True):
            result *= function(x[n:n + d])
            n += d

        return result

    def __repr__(self):
        return (f"{type(self).__name__}(mi={self.multi_index}, "
                f"dims={self.dims_per_function}, functions={self.functions})")


@dataclass(frozen=True)
class _TensorProductGradientBasisFunction:
    r"""
    .. autoattribute:: multi_index
    .. autoattribute:: derivatives
    .. autoattribute:: dims_per_function
    """

    multi_index: tuple[int, ...]
    """A :class:`tuple` used to identify each function in :attr:`functions`
    that is mainly meant for debugging and not used internally.
    """

    derivatives: tuple[tuple[
        Callable[[np.ndarray], np.ndarray | tuple[np.ndarray, ...]],
        ...], ...]
    r"""A :class:`tuple` of :class:`tuple`\ s of callables ``df[i][j]`` that
    evaluate the derivatives of the tensor product. Each ``df[i]`` tuple
    is equivalent to a :class:`_TensorProductBasisFunction` and is used
    to evaluate the derivatives of a single basis function of the tensor
    product. To be specific, a basis function in the tensor product is
    given by

    .. math::

        f(x_1, \dots, x_d) =
                f_1(x_1, \dots, x_{d_1}) \times \cdots \times
                f_n(x_{d_{n - 1}}, \dots, x_{d_n})

    and its derivative with respect to :math:`x_k`, for :math:`k \in
    [d_i, d_{i + 1})` is given by

    .. math::

        \frac{\partial f}{\partial x_k} =
            f_1 \times \cdots \times
            \frac{\partial f_i}{x_k}
            \times \cdots \times
            f_n.

    In our notation, ``df[i]`` gives all the derivatives of :math:`f`
    with respect to :math:`k \in [d_i, d_{i + 1})`. When evaluating
    ``df[i][j]`` can be a function :math:`f_i`, for which the callable
    just returns the function values, or :math:`\partial_k f_i`, for
    which it returns all the derivatives with respect to
    :math:`k \in [d_i, d_{i + 1}]`."""

    dims_per_function: tuple[int, ...] = field(kw_only=True)
    r"""A :class:`tuple` containing the dimensions :math:`(d_1, \dots, d_n)`."""

    def __post_init__(self) -> None:
        assert all(len(self.dims_per_function) == len(df) for df in self.derivatives)

    @property
    def ndim(self) -> int:
        return sum(self.dims_per_function)

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, ...]:
        assert x.shape[0] == self.ndim

        result: list[int | np.ndarray] = [1] * self.ndim
        n = 0
        for ider, derivative in enumerate(self.derivatives):
            f = 0
            for iaxis, function in zip(self.dims_per_function, derivative, strict=True):
                components = function(x[f:f + iaxis])

                if isinstance(components, tuple):
                    # NOTE: this is a derivative evaluation, so it should give
                    # a tuple of derivative wrt to each variable of f_i
                    assert len(components) == self.dims_per_function[ider]
                else:
                    # NOTE: this is a function evaluation, so we distribute
                    # it to all the components of the final derivatives
                    components = (components,) * self.dims_per_function[ider]

                for j, comp in enumerate(components):
                    result[n + j] *= comp

                f += iaxis
            n += self.dims_per_function[ider]

        return cast(tuple[np.ndarray, ...], tuple(result))

    def __repr__(self):
        return (f"{type(self).__name__}(mi={self.multi_index}, "
                f"dims={self.dims_per_function}, derivatives={self.derivatives})")

# }}}


# {{{ conversion to symbolic

def symbolicize_function(
        f: Callable[[RealValueT], RealValueT | tuple[RealValueT, ...]],
        dim: int,
        ref_coord_var_name: str = "r",
        ) -> RealValueT | tuple[RealValueT, ...]:
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

BasisFunctionType = Callable[[np.ndarray], np.ndarray]
BasisGradientType = Callable[[np.ndarray], tuple[np.ndarray, ...]]


class BasisNotOrthonormal(Exception):
    """Exception raised when a :class:`Basis` is expected to be orthonormal."""


class Basis(ABC):
    @abstractmethod
    def orthonormality_weight(self) -> float:
        """
        :raises: :exc:`BasisNotOrthonormal` if the basis does not have
            a weight, i.e. it is not orthogonal.
            For now, only scalar orthonormality weights are supported.
            In the future, this may become a symbolic expression involving
            symbols ``"rstuvw"``.
        """

    @property
    @abstractmethod
    def mode_ids(self) -> tuple[Hashable, ...]:
        """A tuple of of mode (basis function) identifiers, one for
        each basis function. Mode identifiers should generally be viewed
        as opaque. They are hashable. For some bases, they are tuples of
        length matching the number of dimensions and indicating the order
        along each reference axis.
        """

    @property
    @abstractmethod
    def functions(self) -> tuple[BasisFunctionType, ...]:
        """A tuple of (callable) basis functions of length matching
        :attr:`mode_ids`.  Each function takes a vector of :math:`(r, s, t)`
        reference coordinates (depending on dimension) as input.
        """

    @property
    @abstractmethod
    def gradients(self) -> tuple[BasisGradientType, ...]:
        """A tuple of (callable) basis functions of length matching
        :attr:`mode_ids`.  Each function takes a vector of :math:`(r, s, t)`
        reference coordinates (depending on dimension) as input.
        Each function returns a tuple of derivatives, one per reference axis.
        """

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


def zerod_basis(x: np.ndarray) -> np.ndarray:
    assert len(x) == 0
    x_sub = np.ones(x.shape[1:], x.dtype)
    return 1 + x_sub


# {{{ PN bases

def _pkdo_1d(order: tuple[int], r: np.ndarray) -> np.ndarray:
    i, = order
    r0, = r
    return jacobi(0, 0, i, r0)


def _grad_pkdo_1d(order: tuple[int], r: np.ndarray) -> tuple[np.ndarray]:
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
        from pytools import (
            generate_nonnegative_integer_tuples_summing_to_at_most as gnitsam,
        )
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
    def orthonormality_weight(self) -> float:
        raise BasisNotOrthonormal

    @property
    def functions(self):
        return tuple(partial(monomial, mid) for mid in self.mode_ids)

    @property
    def gradients(self):
        return tuple(partial(grad_monomial, mid) for mid in self.mode_ids)


@basis_for_space.register(PN)
def _basis_for_pn(space: PN, shape: Simplex):
    if not isinstance(shape, Simplex):
        raise NotImplementedError((type(space).__name__, type(shape).__name__))

    if space.spatial_dim <= 3:
        return _SimplexONB(space.spatial_dim, space.order)
    else:
        return _SimplexMonomialBasis(space.spatial_dim, space.order)


@orthonormal_basis_for_space.register(PN)
def _orthonormal_basis_for_pn(space: PN, shape: Simplex):
    if not isinstance(shape, Simplex):
        raise NotImplementedError((type(space).__name__, type(shape).__name__))

    return _SimplexONB(space.spatial_dim, space.order)


@monomial_basis_for_space.register(PN)
def _monomial_basis_for_pn(space: PN, shape: Simplex):
    if not isinstance(shape, Simplex):
        raise NotImplementedError((type(space).__name__, type(shape).__name__))

    return _SimplexMonomialBasis(space.spatial_dim, space.order)

# }}}


# {{{ generic tensor product bases

class TensorProductBasis(Basis):
    """Adapts multiple bases into a tensor product basis."""

    def __init__(self,
            bases: Sequence[Basis],
            dims_per_basis: tuple[int, ...] | None = None) -> None:
        """
        :param bases: a sequence of 1D bases used to construct the tensor
            product approximation basis.
        """

        if dims_per_basis is None:
            dims_per_basis = (1,) * len(bases)

        self._bases = tuple(bases)
        self._dims_per_basis = tuple(dims_per_basis)

    def orthonormality_weight(self) -> float:
        orth_weight: float = 1.0
        for b in self._bases:
            orth_weight *= b.orthonormality_weight()
        return orth_weight

    @property
    def bases(self) -> Sequence[Basis]:
        """A sequence of :class:`Basis` objects that are being composed into
        a tensor-product basis in a higher-dimensional space.
        """
        return self._bases

    @property
    def _dim(self):
        return sum(self._dims_per_basis)

    @property
    def _nbases(self):
        return len(self._bases)

    @property
    def _mode_index_tuples(self) -> tuple[tuple[int, ...], ...]:
        from pytools import generate_nonnegative_integer_tuples_below as gnitb

        # ensure that these start numbering (0,0), (1,0), (i.e. x-axis first)
        return tuple(mid[::-1]
                     for mid in gnitb([len(b.functions)
                                       for b in self._bases[::-1]]))

    @property
    def mode_ids(self) -> tuple[Hashable, ...]:
        underlying_mode_id_lists = [basis.mode_ids for basis in self._bases]
        is_all_singletons_with_int = [
                all(isinstance(mid, tuple) and len(mid) == 1
                    and isinstance(mid[0], int)
                    for mid in mode_id_list)
                for mode_id_list in underlying_mode_id_lists]

        def part_flat_tuple(iterable: Iterable[tuple[bool, Hashable]]
                            ) -> tuple[Hashable, ...]:
            result: list[Hashable] = []
            for flatten, item in iterable:
                if flatten:
                    assert isinstance(item, tuple)
                    result.extend(item)
                else:
                    result.append(item)

            return tuple(result)

        return tuple(
                part_flat_tuple((flatten, umid[mid_index_i])
                      for flatten, umid, mid_index_i in zip(
                          is_all_singletons_with_int,
                          underlying_mode_id_lists, mode_index_tuple,
                          strict=True))
                for mode_index_tuple in self._mode_index_tuples)

    @property
    def functions(self) -> tuple[BasisFunctionType, ...]:
        bases = [b.functions for b in self._bases]
        return tuple(
                _TensorProductBasisFunction(
                    mid,
                    tuple(bases[ibasis][mid_i]
                            for ibasis, mid_i in enumerate(mid)),
                    dims_per_function=self._dims_per_basis)
                for mid in self._mode_index_tuples)

    @property
    def gradients(self) -> tuple[BasisGradientType, ...]:
        from pytools import wandering_element
        bases = [b.functions for b in self._bases]
        grad_bases = [b.gradients for b in self._bases]
        func = (bases, grad_bases)
        return tuple(
                _TensorProductGradientBasisFunction(
                    mid,
                    tuple(
                        tuple(func[is_deriv][ibasis][mid_i]
                            for ibasis, (is_deriv, mid_i) in enumerate(
                                zip(deriv_indicator_vec, mid, strict=True)))
                        for deriv_indicator_vec in wandering_element(self._nbases)),
                    dims_per_function=self._dims_per_basis)
                for mid in self._mode_index_tuples)


@orthonormal_basis_for_space.register(TensorProductSpace)
def _orthonormal_basis_for_tp(
        space: TensorProductSpace,
        shape: TensorProductShape):
    if not isinstance(shape, TensorProductShape):
        raise NotImplementedError((type(space).__name__, type(shape).__name__))

    if space.spatial_dim != shape.dim:
        raise ValueError("spatial dimensions of shape and space must match")

    bases = [orthonormal_basis_for_space(b, s)
            for b, s in zip(space.bases, shape.bases, strict=True)]

    return TensorProductBasis(
            bases,
            dims_per_basis=tuple(b.spatial_dim for b in space.bases))


@basis_for_space.register(TensorProductSpace)
def _basis_for_tp(space: TensorProductSpace, shape: TensorProductShape):
    if not isinstance(shape, TensorProductShape):
        raise NotImplementedError((type(space).__name__, type(shape).__name__))

    if space.spatial_dim != shape.dim:
        raise ValueError("spatial dimensions of shape and space must match")

    bases = [basis_for_space(b, s)
        for b, s in zip(space.bases, shape.bases, strict=True)]
    return TensorProductBasis(
            bases,
            dims_per_basis=tuple(b.spatial_dim for b in space.bases))


@monomial_basis_for_space.register(TensorProductSpace)
def _monomial_basis_for_tp(space: TensorProductSpace, shape: TensorProductShape):
    if not isinstance(shape, TensorProductShape):
        raise NotImplementedError((type(space).__name__, type(shape).__name__))

    bases = [
            monomial_basis_for_space(b, s)
            for b, s in zip(space.bases, shape.bases, strict=True)]

    return TensorProductBasis(
            bases,
            dims_per_basis=tuple(b.spatial_dim for b in space.bases))

# }}}

# vim: foldmethod=marker
