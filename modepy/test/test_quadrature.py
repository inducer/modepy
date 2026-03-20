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
from functools import partial
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.linalg as la
import pytest

import modepy as mp
from modepy.quadrature.transplanted import (
    map_identity,
    map_kosloff_tal_ezer,
    map_sausage,
    map_strip,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from modepy.typing import ArrayF


logger = logging.getLogger(__name__)


def test_transformed_quadrature() -> None:
    """Test 1D quadrature on arbitrary intervals"""

    def gaussian_density(x: ArrayF, mu: float, sigma: float) -> ArrayF:
        return (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-np.sum((x - mu) ** 2, axis=0) / (2 * sigma**2))
        )

    from modepy.quadrature import Transformed1DQuadrature
    from modepy.quadrature.jacobi_gauss import LegendreGaussQuadrature

    mu = 17
    sigma = 12
    tq = Transformed1DQuadrature(
        LegendreGaussQuadrature(20, force_dim_axis=True),
        left=mu - 6 * sigma,
        right=mu + 6 * sigma,
    )

    result = tq(lambda x: gaussian_density(x, mu, sigma))
    assert np.abs(result - 1.0) < 1.0e-9


try:
    import scipy  # noqa: F401
except ImportError:
    BACKENDS = [None, "builtin"]
else:
    BACKENDS = [None, "builtin", "scipy"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("quad_cls", [
                             mp.LegendreGaussQuadrature,
                             mp.LegendreGaussLobattoQuadrature,
                         ])
def test_gauss_quadrature(backend: str, quad_cls: type[mp.Quadrature]) -> None:
    for s in range(9 + 1):
        if quad_cls is mp.LegendreGaussLobattoQuadrature and s == 0:
            # no one-node Lobatto rule
            continue

        quad = quad_cls(s, backend=backend, force_dim_axis=True)

        assert quad.nodes.shape[1] == s + 1
        for deg in range(quad.exact_to + 1):
            def f(x: ArrayF) -> ArrayF:
                return np.sum(x**deg, axis=0)  # noqa: B023

            i_f = quad(f)
            i_f_true = 1 / (deg + 1) * (1 - (-1) ** (deg + 1))
            err = abs(i_f - i_f_true)
            assert err < 3.0e-15, (s, deg, err, i_f, i_f_true)


def test_clenshaw_curtis_quadrature() -> None:
    from modepy.quadrature.clenshaw_curtis import ClenshawCurtisQuadrature

    for s in range(1, 9 + 1):
        quad = ClenshawCurtisQuadrature(s, force_dim_axis=True)
        assert quad.nodes.shape[1] == s + 1
        for deg in range(quad.exact_to + 1):
            def f(x: ArrayF) -> ArrayF:
                return x**deg  # noqa: B023

            i_f = quad(f)
            i_f_true = 1 / (deg + 1) * (1 - (-1) ** (deg + 1))
            err = abs(i_f - i_f_true)
            assert err < 2.0e-15, (s, deg, err, i_f, i_f_true)


@pytest.mark.parametrize(
    ("map_fn", "check_exact_weight_sum"),
    [
        (map_identity, True),
        (partial(map_sausage, degree=5), True),
        (partial(map_sausage, degree=9), True),
        (partial(map_sausage, degree=17), True),
        (partial(map_kosloff_tal_ezer, rho=1.4), False),
    ],
)
def test_transplanted_legendre_gauss_quadrature(
    map_fn: Callable[[ArrayF], tuple[ArrayF, ArrayF]],
    check_exact_weight_sum: bool,
) -> None:
    base = mp.LegendreGaussQuadrature(15, force_dim_axis=True)
    transplanted = mp.transplanted_legendre_gauss_quadrature(
        15,
        map_fn,
        force_dim_axis=True,
    )

    base_nodes = cast("ArrayF", np.asarray(base.nodes[0], dtype=np.float64))
    trans_nodes = cast("ArrayF", np.asarray(transplanted.nodes[0], dtype=np.float64))
    base_weights = cast("ArrayF", np.asarray(base.weights, dtype=np.float64))
    trans_weights = cast("ArrayF", np.asarray(transplanted.weights, dtype=np.float64))

    assert transplanted.nodes.shape == base.nodes.shape
    assert transplanted.weights.shape == base.weights.shape

    mapped_nodes, mapped_jacobian = map_fn(base_nodes)

    assert la.norm(trans_nodes - mapped_nodes, np.inf) < 1.0e-14
    assert la.norm(trans_weights - base_weights * mapped_jacobian, np.inf) < 1.0e-14

    # Polynomial maps preserve exactness for constant integrands.
    if check_exact_weight_sum:
        err = abs(float(np.sum(trans_weights)) - 2.0)
        assert err < 1.0e-14


def test_transplanted_strip_map_quadrature() -> None:
    pytest.importorskip("scipy")

    transplanted = mp.transplanted_legendre_gauss_quadrature(
        32,
        partial(map_strip, rho=1.4),
        force_dim_axis=True,
    )

    strip_nodes = cast("ArrayF", np.asarray(transplanted.nodes[0], dtype=np.float64))
    strip_weights = cast("ArrayF", np.asarray(transplanted.weights, dtype=np.float64))

    assert np.all(np.diff(strip_nodes) > 0)
    assert np.all(strip_weights > 0)
    assert abs(float(np.sum(strip_weights)) - 2.0) < 1.0e-9


def test_transplanted_strip_map_rejects_endpoint_rules() -> None:
    pytest.importorskip("scipy")

    with pytest.raises(ValueError, match="interior nodes"):
        mp.transplanted_1d_quadrature(
            mp.ClenshawCurtisQuadrature(5, force_dim_axis=True), map_strip
        )


def test_transplanted_sausage_map_rejects_even_degrees() -> None:
    with pytest.raises(ValueError, match="positive odd degree"):
        mp.transplanted_legendre_gauss_quadrature(
            8,
            partial(map_sausage, degree=4),
            force_dim_axis=True,
        )


def test_transplanted_kte_map_rho_alpha_equivalence() -> None:
    s = np.linspace(-1.0, 1.0, 33)
    rho = 1.4
    alpha = 2.0 / (rho + 1.0 / rho)

    g, gp = map_kosloff_tal_ezer(s, rho=rho)
    g_ref, gp_ref = map_kosloff_tal_ezer(s, alpha=alpha)
    assert la.norm(g - g_ref, np.inf) < 1.0e-15
    assert la.norm(gp - gp_ref, np.inf) < 1.0e-15


def test_transplanted_kte_map_rejects_invalid_parameters() -> None:
    s = np.array([0.0])

    with pytest.raises(ValueError, match="rho must be > 1"):
        map_kosloff_tal_ezer(s, rho=1.0)

    with pytest.raises(ValueError, match="0 < alpha < 1"):
        map_kosloff_tal_ezer(s, alpha=1.0)


@pytest.mark.parametrize("kind", [1, 2])
def test_fejer_quadrature(kind: int) -> None:
    from modepy.quadrature.clenshaw_curtis import FejerQuadrature

    for deg in range(1, 9 + 1):
        s = deg * 3
        quad = FejerQuadrature(s, kind, force_dim_axis=True)

        def f(x: ArrayF) -> ArrayF:
            return x**deg  # noqa: B023

        i_f = quad(f)
        i_f_true = 1 / (deg + 1) * (1 - (-1) ** (deg + 1))
        err = abs(i_f - i_f_true)
        assert err < 2.0e-15, (s, deg, err, i_f, i_f_true)


@pytest.mark.parametrize(("quad_cls", "highest_order"), [
    (mp.XiaoGimbutasSimplexQuadrature, None),
    (mp.JaskowiecSukumarQuadrature, None),
    (mp.VioreanuRokhlinSimplexQuadrature, None),
    (mp.GrundmannMoellerSimplexQuadrature, 3),
    ])
@pytest.mark.parametrize("dim", [2, 3])
def test_simplex_quadrature(quad_cls: type[mp.Quadrature],
                            highest_order: int | None,
                            dim: int) -> None:
    """Check that quadratures on simplices works as advertised"""
    from pytools import (
        generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
    )

    from modepy.tools import Monomial

    order = 1
    while True:
        try:
            quad = quad_cls(order, dim)
        except mp.QuadratureRuleUnavailable:
            print(("UNAVAILABLE", quad_cls, order))
            break

        if isinstance(quad_cls, mp.VioreanuRokhlinSimplexQuadrature):
            assert (quad.weights > 0).all()

        if isinstance(quad_cls, mp.JaskowiecSukumarQuadrature):
            assert (quad.weights > 0).all()

        if 0:
            import matplotlib.pyplot as pt

            pt.plot(quad.nodes[0], quad.nodes[1])
            pt.show()

        print((quad_cls, order, quad.exact_to))
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
@pytest.mark.parametrize(("quad_cls", "max_order"), [
    (mp.WitherdenVincentQuadrature, np.inf),
    (mp.LegendreGaussTensorProductQuadrature, 6),
    ])
def test_hypercube_quadrature(dim: int,
                              quad_cls: type[mp.Quadrature],
                              max_order: float) -> None:
    from pytools import (
        generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
    )

    from modepy.tools import Monomial

    def _check_monomial(quad: mp.Quadrature, comb: Sequence[int]) -> float:
        f = Monomial(comb)
        int_approx = quad(f)
        int_exact = 2.0**dim * f.hypercube_integral()

        error = abs(int_approx - int_exact) / abs(int_exact)
        logger.info(
            "%s: %.5e %.5e / rel error %.5e", comb, int_approx, int_exact, error
        )

        return error

    order = 1
    while order < max_order:
        try:
            quad = quad_cls(order, dim)
        except mp.QuadratureRuleUnavailable:
            logger.info("UNAVAILABLE at order %d", order)
            break

        quad_exact_to = cast("int", quad.exact_to)

        assert np.all(quad.weights > 0)

        logger.info("quadrature: %s %d %d",
                quad_cls.__name__.lower(), order, quad.exact_to)
        for comb in gnitstam(quad.exact_to, dim):
            assert _check_monomial(quad, comb) < 5.0e-15

        comb = (0,) * (dim - 1) + (quad_exact_to + 1,)
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
    mass_weights = cast("ArrayF", mass_weights)
    newton_cotes_weights = cast("ArrayF", newton_cotes_weights)

    assert (
        la.norm(newton_cotes_weights - mass_weights, np.inf)
        / la.norm(newton_cotes_weights, np.inf)
    ) < 1e-13


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
