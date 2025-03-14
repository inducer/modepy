from __future__ import annotations


__copyright__ = "Copyright (C) 2009-2013 Andreas Kloeckner"

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


import logging

import numpy as np
import numpy.linalg as la
import pytest

import modepy as mp


logger = logging.getLogger(__name__)


def test_transformed_quadrature():
    """Test 1D quadrature on arbitrary intervals"""

    def gaussian_density(x, mu, sigma):
        return (
            1 / (sigma * np.sqrt(2*np.pi))
            * np.exp(-np.sum((x-mu)**2, axis=0) / (2 * sigma**2))
        )

    from modepy.quadrature import Transformed1DQuadrature
    from modepy.quadrature.jacobi_gauss import LegendreGaussQuadrature

    mu = 17
    sigma = 12
    tq = Transformed1DQuadrature(
            LegendreGaussQuadrature(20, force_dim_axis=True),
            left=mu - 6*sigma, right=mu + 6*sigma)

    result = tq(lambda x: gaussian_density(x, mu, sigma))
    assert abs(result - 1) < 1.0e-9


try:
    import scipy  # noqa: F401
except ImportError:
    BACKENDS = [None, "builtin"]
else:
    BACKENDS = [None, "builtin", "scipy"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("quad_type", [
                             mp.LegendreGaussQuadrature,
                             mp.LegendreGaussLobattoQuadrature,
                         ])
def test_gauss_quadrature(backend, quad_type):
    for s in range(9 + 1):
        if quad_type == mp.LegendreGaussLobattoQuadrature and s == 0:
            # no one-node Lobatto rule
            continue

        quad = quad_type(s, backend=backend, force_dim_axis=True)

        assert quad.nodes.shape[1] == s+1
        for deg in range(quad.exact_to + 1):
            def f(x):
                return np.sum(x**deg, axis=0)  # noqa: B023

            i_f = quad(f)
            i_f_true = 1 / (deg+1) * (1 - (-1)**(deg + 1))
            err = abs(i_f - i_f_true)
            assert err < 3.0e-15, (s, deg, err, i_f, i_f_true)


def test_clenshaw_curtis_quadrature() -> None:
    from modepy.quadrature.clenshaw_curtis import ClenshawCurtisQuadrature

    for s in range(1, 9 + 1):
        quad = ClenshawCurtisQuadrature(s, force_dim_axis=True)
        assert quad.nodes.shape[1] == s+1
        for deg in range(quad.exact_to + 1):
            def f(x):
                return x**deg  # noqa: B023

            i_f = quad(f)
            i_f_true = 1 / (deg+1) * (1 - (-1)**(deg + 1))
            err = abs(i_f - i_f_true)
            assert err < 2.0e-15, (s, deg, err, i_f, i_f_true)


@pytest.mark.parametrize("kind", [1, 2])
def test_fejer_quadrature(kind: int) -> None:
    from modepy.quadrature.clenshaw_curtis import FejerQuadrature

    for deg in range(1, 9 + 1):
        s = deg * 3
        quad = FejerQuadrature(s, kind, force_dim_axis=True)

        def f(x):
            return x**deg  # noqa: B023

        i_f = quad(f)
        i_f_true = 1 / (deg+1) * (1 - (-1)**(deg + 1))
        err = abs(i_f - i_f_true)
        assert err < 2.0e-15, (s, deg, err, i_f, i_f_true)


@pytest.mark.parametrize(("quad_class", "highest_order"), [
    (mp.XiaoGimbutasSimplexQuadrature, None),
    (mp.JaskowiecSukumarQuadrature, None),
    (mp.VioreanuRokhlinSimplexQuadrature, None),
    (mp.GrundmannMoellerSimplexQuadrature, 3),
    ])
@pytest.mark.parametrize("dim", [2, 3])
def test_simplex_quadrature(quad_class, highest_order, dim) -> None:
    """Check that quadratures on simplices works as advertised"""
    from pytools import (
        generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
    )

    from modepy.tools import Monomial

    order = 1
    while True:
        try:
            quad = quad_class(order, dim)
        except mp.QuadratureRuleUnavailable:
            print(("UNAVAILABLE", quad_class, order))
            break

        if isinstance(quad_class, mp.VioreanuRokhlinSimplexQuadrature):
            assert (quad.weights > 0).all()

        if isinstance(quad_class, mp.JaskowiecSukumarQuadrature):
            assert (quad.weights > 0).all()

        if 0:
            import matplotlib.pyplot as pt
            pt.plot(quad.nodes[0], quad.nodes[1])
            pt.show()

        print((quad_class, order, quad.exact_to))
        for comb in gnitstam(quad.exact_to, dim):
            f = Monomial(comb)
            i_f = quad(f)
            ref = f.simplex_integral()
            err = abs(i_f - ref)
            print(order, repr(f), err)
            assert err < 6e-15, (err, comb, i_f, ref)

        order += 1
        if highest_order is not None and order >= highest_order:
            break


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(("quad_class", "max_order"), [
    (mp.WitherdenVincentQuadrature, np.inf),
    (mp.LegendreGaussTensorProductQuadrature, 6),
    ])
def test_hypercube_quadrature(dim, quad_class, max_order):
    from pytools import (
        generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
    )

    from modepy.tools import Monomial

    def _check_monomial(quad, comb):
        f = Monomial(comb)
        int_approx = quad(f)
        int_exact = 2**dim * f.hypercube_integral()

        error = abs(int_approx - int_exact) / abs(int_exact)
        logger.info("%s: %.5e %.5e / rel error %.5e",
                comb, int_approx, int_exact, error)

        return error

    order = 1
    while order < max_order:
        try:
            quad = quad_class(order, dim)
        except mp.QuadratureRuleUnavailable:
            logger.info("UNAVAILABLE at order %d", order)
            break

        assert np.all(quad.weights > 0)

        logger.info("quadrature: %s %d %d",
                quad_class.__name__.lower(), order, quad.exact_to)
        for comb in gnitstam(quad.exact_to, dim):
            assert _check_monomial(quad, comb) < 5.0e-15

        comb = (0,) * (dim - 1) + (quad.exact_to + 1,)
        assert _check_monomial(quad, comb) > 5.0e-15

        order += 2


@pytest.mark.parametrize("shape", [mp.Simplex(1), mp.Simplex(2), mp.Simplex(3)])
@pytest.mark.parametrize("order", [1, 3, 8])
def test_mass_quadrature_is_newton_cotes(shape: mp.Shape, order: int) -> None:
    space = mp.space_for_shape(shape, order)
    basis = mp.basis_for_space(space, shape)

    nodes = mp.equispaced_nodes_for_space(space, shape)
    mass_mat = mp.mass_matrix(basis, nodes)
    mass_weights = np.ones(len(mass_mat)) @ mass_mat

    vdm = mp.vandermonde(basis.functions, nodes)
    assert basis.orthonormality_weight() == 1

    from math import factorial
    shape_volume = 2**shape.dim / factorial(shape.dim)

    # integrals are orthogonal to the constant
    integrals = np.zeros(len(basis.functions))
    # boldly assume that the first basis function is the constant
    integrals[0] = shape_volume * basis.functions[0](np.zeros(shape.dim))

    newton_cotes_weights = la.solve(vdm.T, integrals)

    assert (la.norm(newton_cotes_weights - mass_weights, np.inf)
            / la.norm(newton_cotes_weights, np.inf)) < 1e-13


# You can test individual routines by typing
# $ python test_quadrature.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
