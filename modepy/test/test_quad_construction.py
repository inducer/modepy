from __future__ import annotations


__copyright__ = "Copyright (C) 2024 University of Illinois Board of Trustees"

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


from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la

import modepy as mp
import modepy.quadrature.construction as constr


if TYPE_CHECKING:
    from modepy.typing import NodalFunction


def test_quad_finder_lincomb() -> None:
    basis = constr.adapt_2d_integrands_to_complex_arg(
        mp.monomial_basis_for_space(
            mp.PN(2, order=5),
            mp.Simplex(2)).functions,
        )

    quad = mp.XiaoGimbutasSimplexQuadrature(20, 2)
    base_quad = mp.Quadrature(
        nodes=quad.nodes[0] + quad.nodes[1]*1j,
        weights=quad.weights,
    )

    n = len(basis)
    rng = np.random.default_rng(15)

    m1 = rng.uniform(size=(n, n))
    b1 = [constr.linearly_combine(m1[i], basis) for i in range(n)]
    b1_ref = [constr.LinearCombinationIntegrand(m1[i], basis) for i in range(n)]

    for test_f, ref_f in zip(b1, b1_ref, strict=True):
        assert la.norm(test_f(base_quad.nodes) - ref_f(base_quad.nodes), 2) < 1e-15

    m2 = rng.uniform(size=(n, n))
    b2 = [constr.linearly_combine(m2[i], b1) for i in range(n)]
    b2_ref = [constr.LinearCombinationIntegrand(m2[i], b1_ref) for i in range(n)]

    for test_f in b2:
        assert isinstance(test_f, constr.LinearCombinationIntegrand)
        assert all(not isinstance(subf, constr.LinearCombinationIntegrand)
                   for subf in test_f.functions)

    for test_f, ref_f in zip(b2, b2_ref, strict=True):
        assert la.norm(test_f(base_quad.nodes) - ref_f(base_quad.nodes), 2) < 1e-13


def test_orthogonalize_basis() -> None:
    orig_basis = mp.monomial_basis_for_space(mp.PN(2, order=7), mp.Simplex(2))
    basis = constr.adapt_2d_integrands_to_complex_arg(orig_basis.functions)

    quad = mp.XiaoGimbutasSimplexQuadrature(20, 2)
    base_quad = mp.Quadrature(
        nodes=quad.nodes[0] + quad.nodes[1]*1j,
        weights=quad.weights,
    )

    def integrate(integrand: NodalFunction) -> np.floating:
        res = base_quad(integrand)
        assert isinstance(res, np.inexact)
        return res

    onb = constr.orthogonalize_basis(integrate, basis)
    onb = constr.orthogonalize_basis(integrate, onb)

    eye = constr._mass_matrix(integrate, onb)
    assert la.norm(eye - np.eye(len(onb))) < 2e-12


def test_guess_nodes_vr() -> None:
    order = 7
    shape = mp.Simplex(2)
    orig_basis = mp.orthonormal_basis_for_space(
            mp.PN(2, order=order),
            shape)
    basis = constr.adapt_2d_integrands_to_complex_arg(orig_basis.functions)

    quad = mp.XiaoGimbutasSimplexQuadrature(20, 2)
    base_quad = mp.Quadrature(
        nodes=quad.nodes[0] + quad.nodes[1]*1j,
        weights=quad.weights,
    )

    def integrate(integrand: NodalFunction) -> np.floating:
        res = base_quad(integrand)
        assert isinstance(res, np.inexact)
        return res

    onb = constr.orthogonalize_basis(integrate, basis)

    eye = constr._mass_matrix(integrate, onb)
    assert la.norm(eye - np.eye(len(onb))) < 1e-14

    nodes_complex = constr.guess_nodes_vioreanu_rokhlin(integrate, onb)
    nodes = np.array([nodes_complex.real, nodes_complex.imag])

    from modepy.tools import estimate_lebesgue_constant
    assert estimate_lebesgue_constant(order, nodes, shape) < 32

    if 0:
        import matplotlib.pyplot as plt
        plt.plot(nodes[0], nodes[1], "x")
        plt.gca().set_aspect("equal")
        plt.show()


def test_undetermined_coeffs() -> None:
    order = 7
    space = mp.PN(2, order=order)
    shape = mp.Simplex(2)
    basis = mp.orthonormal_basis_for_space(space, shape)
    nodes = mp.edge_clustered_nodes_for_space(space, shape)

    ref_quad = mp.XiaoGimbutasSimplexQuadrature(20, 2)

    ref_integrals = np.array([ref_quad(f) for f in basis.functions])
    weights = constr.find_weights_undetermined_coefficients(
                                    basis.functions, nodes, ref_integrals)
    quad = mp.Quadrature(nodes, weights)

    from pytools import (
        generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
    )

    from modepy.tools import Monomial
    for comb in gnitstam(order, 2):
        f = Monomial(comb)
        i_f = quad(f)
        ref = f.simplex_integral()
        err = abs(i_f - ref)
        print(order, repr(f), err)
        assert err < 2e-15, (err, comb, i_f, ref)


def test_quad_residual_derivatives() -> None:
    order = 7
    shape = mp.Simplex(2)
    onb = mp.orthonormal_basis_for_space(
            mp.PN(2, order=order),
            shape)
    complex_onb = constr.adapt_2d_integrands_to_complex_arg(onb.functions)

    base_quad = mp.XiaoGimbutasSimplexQuadrature(20, 2)
    base_quad_complex = mp.Quadrature(
        nodes=base_quad.nodes[0] + base_quad.nodes[1]*1j,
        weights=base_quad.weights,
    )

    def integrate(integrand: NodalFunction) -> np.floating:
        res = base_quad_complex(integrand)
        assert isinstance(res, np.inexact)
        return res

    eye = constr._mass_matrix(integrate, complex_onb)
    assert la.norm(eye - np.eye(len(complex_onb))) < 7e-14

    nodes_complex = constr.guess_nodes_vioreanu_rokhlin(integrate, complex_onb)
    nodes = np.array([nodes_complex.real, nodes_complex.imag])

    ref_integrals = np.array([base_quad(f) for f in onb.functions])
    weights = constr.find_weights_undetermined_coefficients(
            onb.functions, nodes, ref_integrals)

    from pytools import (
        generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
    )

    from modepy.tools import Monomial

    quad = mp.Quadrature(nodes, weights)

    for comb in gnitstam(order, 2):
        f = Monomial(comb)
        assert abs(quad(f) - f.simplex_integral()) < 2e-15

    all_derivatives = [onb.derivatives(iaxis) for iaxis in range(shape.dim)]

    qrj = constr.quad_residual_and_jacobian(
                    nodes, weights, onb.functions, all_derivatives, ref_integrals)

    rng = np.random.default_rng(17)

    # check weight derivatives
    w_direction = rng.uniform(size=weights.shape)
    for h in [0.1]:
        qrj_offset = constr.quad_residual_and_jacobian(
                        nodes, weights + h * w_direction,
                        onb.functions, all_derivatives,
                        ref_integrals)

        err = la.norm(
            (qrj_offset.residual - qrj.residual) / h
            - qrj.dresid_dweights @ w_direction, 2)
        # Residual is linear in the weights
        assert err < 1e-12

    # check node derivatives
    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    n_direction = rng.uniform(size=nodes.shape)
    for h in [10**-i for i in range(1, 4)]:
        qrj_offset = constr.quad_residual_and_jacobian(
                        nodes + h * n_direction, weights,
                        onb.functions, all_derivatives,
                        ref_integrals)

        err = la.norm(
            (qrj_offset.residual - qrj.residual) / h
            - qrj.dresid_dnodes @ n_direction.reshape(-1), 2)
        eoc_rec.add_data_point(h, float(err))

    print(eoc_rec)
    assert eoc_rec.order_estimate() >= 0.95


def test_quad_gauss_newton() -> None:
    order = 7
    shape = mp.Simplex(2)
    onb = mp.orthonormal_basis_for_space(
            mp.PN(2, order=order),
            shape)
    complex_onb = constr.adapt_2d_integrands_to_complex_arg(onb.functions)

    base_quad = mp.XiaoGimbutasSimplexQuadrature(20, 2)
    base_quad_complex = mp.Quadrature(
        nodes=base_quad.nodes[0] + base_quad.nodes[1]*1j,
        weights=base_quad.weights,
    )

    def integrate(integrand: NodalFunction) -> np.floating:
        res = base_quad_complex(integrand)
        assert isinstance(res, np.inexact)
        return res

    eye = constr._mass_matrix(integrate, complex_onb)
    assert la.norm(eye - np.eye(len(complex_onb))) < 7e-14

    nodes_complex = constr.guess_nodes_vioreanu_rokhlin(integrate, complex_onb)
    nodes = np.array([nodes_complex.real, nodes_complex.imag])

    integrands = mp.orthonormal_basis_for_space(
            mp.PN(2, order=11),
            shape)

    ref_integrals = np.array([base_quad(f) for f in integrands.functions])
    weights = constr.find_weights_undetermined_coefficients(
            integrands.functions, nodes, ref_integrals)

    all_derivatives = [integrands.derivatives(iaxis) for iaxis in range(shape.dim)]

    for _ in range(8):
        qrj = constr.quad_residual_and_jacobian(
            nodes, weights, integrands.functions, all_derivatives, ref_integrals)
        weights_inc, nodes_inc = constr.quad_gauss_newton_increment(qrj)
        weights += weights_inc
        nodes += nodes_inc

        resid_norm = la.norm(qrj.residual)

    assert resid_norm < 1e-14
