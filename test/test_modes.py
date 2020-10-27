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


import numpy as np
import numpy.linalg as la
import pytest
import modepy.modes as m

import logging
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(("alpha", "beta", "ebound"), [
    (0, 0, 5e-14),              # Gauss-Legendre
    (-0.5, -0.5, 6e-15),        # Chebyshev-Gauss (first kind)
    (0.5, 0.5, 6e-15),          # Chebyshev-Gauss (second kind)
    (1, 0, 4e-14),
    (3, 2, 3e-14),
    (0, 2, 3e-13),
    (5, 0, 3e-13),
    (3, 4, 1e-14)
    ])
def test_orthonormality_jacobi_1d(alpha, beta, ebound):
    """Verify that the Jacobi polymials are orthogonal in 1D."""
    from modepy.quadrature.jacobi_gauss import JacobiGaussQuadrature

    max_n = 10
    quad = JacobiGaussQuadrature(alpha, beta, 4*max_n)

    from functools import partial
    jac_f = [partial(m.jacobi, alpha, beta, n) for n in range(max_n)]
    maxerr = 0

    for i, fi in enumerate(jac_f):
        for j, fj in enumerate(jac_f):
            result = quad(lambda x: fi(x)*fj(x))
            true_result = 1.0 if i == j else 0.0

            err = abs(result-true_result)
            maxerr = max(maxerr, err)
            if abs(result - true_result) > ebound:
                logger.error("[FAILED] (%g, %g): (%d, %d) error %.5e",
                        alpha, beta, i, j, abs(result - true_result))

            assert abs(result-true_result) < ebound


@pytest.mark.parametrize(("order", "ebound"), [
    (1, 2e-15),
    (2, 5e-15),
    (3, 1e-14),
    # (4, 3e-14),
    # (7, 3e-14),
    # (9, 2e-13),
    ])
@pytest.mark.parametrize("dims", [2, 3])
def test_pkdo_orthogonality(dims, order, ebound):
    """Test orthogonality of simplicial bases using Grundmann-Moeller cubature."""

    from modepy.quadrature.grundmann_moeller import GrundmannMoellerSimplexQuadrature
    from modepy.modes import simplex_onb

    cub = GrundmannMoellerSimplexQuadrature(order, dims)
    basis = simplex_onb(dims, order)

    maxerr = 0
    for i, f in enumerate(basis):
        for j, g in enumerate(basis):
            if i == j:
                true_result = 1
            else:
                true_result = 0
            result = cub(lambda x: f(x)*g(x))
            err = abs(result-true_result)
            print((maxerr, err))
            maxerr = max(maxerr, err)
            if err > ebound:
                print("bad", order, i, j, err)
            assert err < ebound
    # print(order, maxerr)


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("order", [5, 8])
@pytest.mark.parametrize(("eltype", "basis_getter", "grad_basis_getter"), [
    ("simplex", m.simplex_onb, m.grad_simplex_onb),
    ("simplex", m.simplex_monomial_basis, m.grad_simplex_monomial_basis),
    ("tensor", m.legendre_tensor_product_basis, m.grad_legendre_tensor_product_basis)
    ])
def test_basis_grad(dims, order, eltype, basis_getter, grad_basis_getter):
    """Do a simplistic FD-style check on the gradients of the basis."""

    h = 1.0e-4
    if eltype == "simplex" and order == 8 and dims == 3:
        factor = 3.0
    else:
        factor = 1.0

    if eltype == "simplex":
        from modepy.tools import \
                pick_random_simplex_unit_coordinate as pick_random_unit_coordinate
    elif eltype == "tensor":
        from modepy.tools import \
                pick_random_hypercube_unit_coordinate as pick_random_unit_coordinate
    else:
        raise ValueError(f"unknown element type: {eltype}")

    from random import Random
    rng = Random(17)

    from pytools import wandering_element
    for i_bf, (bf, gradbf) in enumerate(zip(
                basis_getter(dims, order),
                grad_basis_getter(dims, order),
                )):
        for i in range(10):
            r = pick_random_unit_coordinate(rng, dims)

            gradbf_v = np.array(gradbf(r))
            gradbf_v_num = np.array([
                (bf(r+h*unit) - bf(r-h*unit))/(2*h)
                for unit_tuple in wandering_element(dims)
                for unit in (np.array(unit_tuple),)
                ])

            err = la.norm(gradbf_v_num - gradbf_v)
            logger.info("error: %.5", err)
            assert err < factor * h, (err, i_bf)


# You can test individual routines by typing
# $ python test_modes.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
