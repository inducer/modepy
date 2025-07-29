"""
.. versionadded:: 2025.2

.. autoclass:: LinearCombinationIntegrand
.. autofunction:: linearly_combine
.. autofunction:: orthogonalize_basis
.. autofunction:: adapt_2d_integrands_to_complex_arg
.. autofunction:: guess_nodes_vioreanu_rokhlin
.. autofunction:: find_weights_undetermined_coefficients
.. autoclass:: QuadratureResidualJacobian
.. autofunction:: quad_residual_and_jacobian
.. autofunction:: quad_gauss_newton_increment
"""

from __future__ import annotations


__copyright__ = """
Copyright (C)  2024 University of Illinois Board of Trustees
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
import operator
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, cast
from warnings import warn

import numpy as np
import numpy.linalg as la


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from modepy.typing import ArrayF, NodalFunction


@dataclass(frozen=True)
class _ProductIntegrand:
    functions: Sequence[NodalFunction]

    def __call__(self, points: ArrayF) -> ArrayF:
        return reduce(operator.mul, (f(points) for f in self.functions))


@dataclass(frozen=True)
class _ConjugateIntegrand:
    function: NodalFunction

    def __call__(self, points: ArrayF) -> ArrayF:
        return self.function(points).conj()


def _identity_integrand(points: ArrayF) -> ArrayF:
    return points


@dataclass(frozen=True)
class LinearCombinationIntegrand:
    """
    .. note::

        New linear combinations should be created by :func:`linearly_combine`,
        which flattens nested combinations by taking advantage of associativity,
        substantially decreasing evaluation cost.

    .. autoattribute:: coefficients
    .. autoattribute:: functions
    .. automethod:: __call__
    """
    coefficients: ArrayF
    functions: Sequence[NodalFunction]

    def __post_init__(self):
        assert len(self.coefficients) == len(self.functions)

    def __call__(self, points: ArrayF) -> ArrayF:
        """Evaluate the linear combination of :attr:`functions` with
        :attr:`coefficients` at *points*.
        """
        return sum(
            (coeff * func(points)
                for coeff, func in zip(self.coefficients, self.functions, strict=True)),
            np.zeros(()))


def linearly_combine(
            coefficients: ArrayF,
            functions: Sequence[NodalFunction]
        ) -> LinearCombinationIntegrand:
    r"""
    Returns a linear combination of *functions* with coefficients.
    If *functions* are themselves :class:`LinearCombinationIntegrand`\ s,
    and all *functions* use a consistent basis, then the resulting
    :class:`LinearCombinationIntegrand` will be flattened via associativity,
    leading to faster execution time.
    """

    lcfunctions = [
        f for f in functions
        if isinstance(f, LinearCombinationIntegrand)
    ]
    if len(lcfunctions) != len(functions):
        if lcfunctions:
            warn("Only some functions passed to linearly_combine were themselves "
                 "linear combination. This case is unhandled."
                 "Returning an un-flattened linear combination.",
                 stacklevel=2)
        return LinearCombinationIntegrand(coefficients, functions)

    basis: list[NodalFunction] = []
    n = len(lcfunctions)
    matrix = np.zeros((n, n), dtype=np.complex128)
    for i, f in enumerate(lcfunctions):
        ncommon = min(len(basis), len(f.functions))
        if basis[:ncommon]  != f.functions[:ncommon]:
            warn("Not all functions passed to linearly_combine used the "
                 "same basis. Returning an un-flattened linear combination instead.",
                 stacklevel=2)

            return LinearCombinationIntegrand(coefficients, functions)

        basis.extend(f.functions[ncommon:])

        ncoeff = len(f.coefficients)
        matrix[i, :ncoeff] = f.coefficients

    # cast because ArrayF is mismatched
    return LinearCombinationIntegrand(cast("ArrayF", coefficients @ matrix), basis)


def _mass_matrix(
            integrate: Callable[[NodalFunction], np.floating],
            basis: Sequence[NodalFunction],
        ) -> ArrayF:
    n = len(basis)
    integrals = [
        [
            integrate(_ProductIntegrand((basis[i], _ConjugateIntegrand(basis[j]))))
            for j in range(i+1)
            ]
        for i in range(n)
    ]

    # Let np.array deal with figuring out the result dtype
    return np.array([
        # Fill out the full matrix from the triangle computed above
        [integrals[i][j] if i >= j else integrals[j][i] for j in range(n)]
        for i in range(n)
    ])


def orthogonalize_basis(
            integrate: Callable[[NodalFunction], np.floating],
            basis: Sequence[NodalFunction],
        ) -> Sequence[LinearCombinationIntegrand]:
    r"""
    Let :math:`\Omega\subset\mathbb C^n` be a domain. (Domains
    over the reals are allowable as well.) Returns linear combinations
    of functions in *basis* that is orthogonal under the (complex-valued)
    :math:`L^2` inner product induced by *integrate*.

    :arg integrate: Computes an integral of the passed integrand over
        :math:`\Omega`. Must use integration nodes compatible with
        *basis*.
    :arg basis: A sequence of functions that accept an array of nodes
        of shape either ``(n, nnodes)`` or ``(nnodes,)`` and return
        an array of shape ``(nnodes,)`` with the value of the
        basis function at the node.
    """
    n = len(basis)
    mass_mat = _mass_matrix(integrate, basis)

    l_factor = la.cholesky(mass_mat)

    try:
        from scipy.linalg import solve_triangular
    except ImportError:
        # *shrug* I guess? The triangular version is also O(n**3).
        l_inv = la.inv(l_factor)
    else:
        l_inv = solve_triangular(l_factor, np.eye(n), lower=True)

    assert la.norm(np.triu(l_inv, 1), "fro") < 1e-14

    return [
        linearly_combine(l_inv[i, :i+1], basis[:i+1])
        for i in range(n)
    ]


@dataclass(frozen=True)
class _ComplexToNDAdapter:
    function: NodalFunction

    def __call__(self, points: ArrayF) -> ArrayF:
        rpoints  = np.array([points.real, points.imag])
        return self.function(rpoints)


def adapt_2d_integrands_to_complex_arg(
            functions: Sequence[NodalFunction]
        ) -> Sequence[NodalFunction]:
    r"""Converts a list of :data:`NodalFunction`\ s taking nodes in real-valued
    arrays of shape ``(2, nnodes)`` to ones accepting a single
    complex-valued array of shape ``(nnodes,)``. See
    :func:`guess_nodes_vioreanu_rokhlin`
    for the main intended use case.
    """
    return [_ComplexToNDAdapter(f) for f in functions]


def guess_nodes_vioreanu_rokhlin(
            integrate: Callable[[NodalFunction], np.floating],
            onb: Sequence[NodalFunction],
        ) -> ArrayF:
    r"""
    Let :math:`\Omega\subset\mathbb C` be a convex domain.
    Finds interpolation nodes for :math:`\Omega` based on the
    multiplication-operator technique in
    [Vioreanu2011]_. May be useful as an initial guess for a Gauss-Newton
    process to find new quadrature rules, as driven by
    :func:`quad_residual_and_jacobian`.

    :arg integrate: Must accurately integrate a product of two functions from *onb* and
        a degree-1 monomial over :math:`\Omega`.
    :arg onb: An orthonormal basis of functions. Each function takes
        in an array of (complex) nodes and returns an array
        of the same shape containing the function values.
        Functions are expected to be orthogonal with respect to
        the complex-valued inner product.

        Integrands accepting real-valued node coordinates in two dimensions
        (shaped ``(2, nnodes)`` may be converted to functions acceptable by
        this function via :func:`adapt_2d_integrands_to_complex_arg`.)
    :returns: An array of shape ``(len(onb),)`` containing nodes, complex-valued.

    .. note::

        This function is based on complex arithmetic and thus only
        usable for domains $\Omega$ with one and two (real-valued)
        dimensions.

    .. note::

        (Empirically) it seems acceptable for the functions
        in *onb* to be exclusively real-valued, i.e., particularly,
        no assumptions on complex differentiability are needed.

    .. note::

        As noted in [Vioreanu2011]_, the returned nodes should obey
        the symmetry of the domain.

    """
    n = len(onb)
    mat = np.array([
        [
            integrate(
                _ProductIntegrand((
                    _identity_integrand,
                    onb[i],
                    _ConjugateIntegrand(onb[j]))))
            for j in range(n)
        ]
        for i in range(n)
    ])

    return la.eigvals(mat)


def find_weights_undetermined_coefficients(
            integrands: Sequence[NodalFunction],
            nodes: ArrayF,
            reference_integrals: ArrayF,
        ) -> ArrayF:
    """
    Uses the method of undetermined coefficients to find weights
    for a quadrature rule using *nodes*, for the provided
    *integrands*.

    :arg integrands: A sequence of functions that accept an array of nodes
        of shape either ``(ndim, nnodes)`` or ``(nnodes,)`` and return
        an array of shape ``(nnodes,)`` with the value of the
        basis function at the node.
    :arg nodes: An array with shape ``(ndim, nnodes)`` or ``(nnodes,)``, real-valued.
        Must be compatible with *integrands*.
    :arg reference_integrals: An array with shape ``(len(integrands),)``

    .. note::

        This function also supports overdetermined systems. In that case, it
        will provide the least squares solution.


    .. note::

        Unlike :func:`guess_nodes_vioreanu_rokhlin`, domains of any dimensionality
        are allowed.
    """

    if len(reference_integrals) != len(integrands):
        raise ValueError(
                "number of integrands must match number of reference integrals")
    if len(reference_integrals) < len(nodes):
        from warnings import warn
        warn("Underdetermined quadrature system", stacklevel=2)

    from modepy import vandermonde
    return la.lstsq(vandermonde(integrands, nodes).T, reference_integrals)[0]


@dataclass(frozen=True)
class QuadratureResidualJacobian:
    """
    Contains residual value and Jacobian for a quadrature residual, see
    :func:`quad_residual_and_jacobian`. May be reused for residuals
    that include additional terms, such as those that penalize asymmetry
    of the noddes.

    .. autoattribute:: residual
    .. autoattribute:: dresid_dweights
    .. autoattribute:: dresid_dnodes
    """

    residual: ArrayF
    """Shaped ``(nintegrands,)``."""

    dresid_dweights: ArrayF
    """Shaped ``(nintegrands, nnodes)``."""

    dresid_dnodes: ArrayF
    """Shaped ``(nintegrands, ndim*nnodes)``."""


def quad_residual_and_jacobian(
            nodes: ArrayF,
            weights: ArrayF,
            integrands: Sequence[NodalFunction],
            integrand_derivatives: Sequence[Sequence[NodalFunction]],
            reference_integrals: ArrayF,
        ) -> QuadratureResidualJacobian:
    r"""
    Computes the residual and Jacobian of the objective function

    .. math::

        \begin{bmatrix}
            \sum_{j=0}^{\text{nnodes-1}} \psi_0(\boldsymbol x_j) w_j - I_0\\
            \vdots\\
            \sum_{j=0}^{\text{nnodes-1}} \psi_{\text{nintegrands-1}}
              (\boldsymbol x_j) w_j - I_{\text{nintegrands}-1}
        \end{bmatrix}.

    Typically used with :meth:`quad_gauss_newton_increment`
    to drive an iterative process for finding nodes and weights of a quadrature
    rule.

    An initial guess for the nodes may be obtained via
    :func:`guess_nodes_vioreanu_rokhlin`,
    and an initial guess for the weights (given nodes) may be found
    via :func:`find_weights_undetermined_coefficients`.

    .. note::

        Unlike :func:`guess_nodes_vioreanu_rokhlin`, domains of any dimensionality
        are allowed.

    :arg nodes: An array with shape ``(ndim, nnodes)`` or ``(nnodes,)``, real-valued.
        Must be compatible with *integrands*.
    :arg weights: An array with shape ``(nnodes,)``, real-valued
    :arg integrands: A sequence of functions that accept an array of nodes
        of shape either ``(ndim, nnodes)`` or ``(nnodes,)`` and return
        an array of shape ``(nnodes,)`` with the value of the
        basis function at the node.
    :arg integrand_derivatives: Derivatives of *integrands* along the
        coordinate axes, one list of
        functions per axis, with each sub-list matching *integrands*
        in structure.
    :arg reference_integrals: Integrals of *integrands*, shaped ``(len(integrands),)``.
    """
    nintegrands = len(integrands)
    ndim, _nnodes = nodes.shape

    if __debug__:
        if len(integrand_derivatives) != ndim:
            raise ValueError("sequence of integrand derivatives must have ndim entries")

        for iaxis, ax_derivatives in enumerate(integrand_derivatives):
            if len(ax_derivatives) != nintegrands:
                raise ValueError(
                    "number of integrands must match number of integrand"
                    f"along axis {iaxis} (0-based)")

    from modepy import vandermonde

    vdm_t = vandermonde(integrands, nodes).T
    residual = vdm_t @ weights - reference_integrals

    dresid_dnodes = np.hstack([
        vandermonde(ax_derivatives, nodes).T * weights
        for ax_derivatives in integrand_derivatives
        ])

    return QuadratureResidualJacobian(
        residual=residual,
        dresid_dweights=vdm_t,
        dresid_dnodes=dresid_dnodes,
    )


def quad_gauss_newton_increment(
            qrj: QuadratureResidualJacobian
        ) -> tuple[ArrayF, ArrayF]:
    """Return the Gauss-Newton increment based on the residual and Jacobian
    (see :func:`quad_residual_and_jacobian`),
    separated into the weight increment and the nodes increment,
    in the two entries of the returned tuple. The nodes increment
    has shape ``(ndim, nnodes)``.

    .. note::

        [Vioreanu2011]_ suggests that step size control should be used
        in the Gauss-Newton process. No notion of step size control
        is included in the returned increments.
    """

    full_jacobian = np.hstack([qrj.dresid_dweights, qrj.dresid_dnodes])
    full_increment = -la.lstsq(full_jacobian, qrj.residual)[0]

    _nintegrands, nnodes = qrj.dresid_dweights.shape

    return full_increment[:nnodes], full_increment[nnodes:].reshape(-1, nnodes)
