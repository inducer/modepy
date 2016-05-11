from __future__ import division, absolute_import, print_function

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

from six.moves import range, zip

import numpy as np
import numpy.linalg as la
import pytest
import modepy.modes as m


@pytest.mark.parametrize(("alpha", "beta", "ebound"), [
    (0, 0, 5e-14),
    (1, 0, 4e-14),
    (3, 2, 3e-14),
    (0, 2, 3e-13),
    (5, 0, 3e-13),
    (3, 4, 1e-14)
    ])
def test_orthonormality_jacobi_1d(alpha, beta, ebound):
    """Verify that the Jacobi polymials are orthogonal in 1D."""
    from modepy.modes import jacobi
    from modepy.quadrature.jacobi_gauss import LegendreGaussQuadrature

    max_n = 10
    quad = LegendreGaussQuadrature(4*max_n)  # overkill...

    class WeightFunction:
        def __init__(self, alpha, beta):
            self.alpha = alpha
            self.beta = beta

        def __call__(self, x):
            return (1-x)**self.alpha * (1+x)**self.beta

    from functools import partial
    jac_f = [partial(jacobi, alpha, beta, n) for n in range(max_n)]
    wf = WeightFunction(alpha, beta)
    maxerr = 0

    for i, fi in enumerate(jac_f):
        for j, fj in enumerate(jac_f):
            result = quad(lambda x: wf(x)*fi(x)*fj(x))

            if i == j:
                true_result = 1
            else:
                true_result = 0
            err = abs(result-true_result)
            maxerr = max(maxerr, err)
            if abs(result-true_result) > ebound:
                print(("bad", alpha, beta, i, j, abs(result-true_result)))

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
@pytest.mark.parametrize(("basis_getter", "grad_basis_getter"), [
    (m.simplex_onb, m.grad_simplex_onb),
    (m.simplex_monomial_basis, m.grad_simplex_monomial_basis),
    ])
def test_simplex_basis_grad(dims, order, basis_getter, grad_basis_getter):
    """Do a simplistic FD-style check on the gradients of the basis."""

    if dims == 3:
        err_factor = 3
    else:
        err_factor = 1

    from random import Random
    rng = Random(17)

    from modepy.tools import pick_random_simplex_unit_coordinate
    for i_bf, (bf, gradbf) in enumerate(zip(
                basis_getter(dims, order),
                grad_basis_getter(dims, order),
                )):
        for i in range(10):
            r = pick_random_simplex_unit_coordinate(rng, dims)

            from pytools import wandering_element
            h = 1e-4
            gradbf_v = np.array(gradbf(r))
            approx_gradbf_v = np.array([
                (bf(r+h*unit) - bf(r-h*unit))/(2*h)
                for unit in [np.array(unit) for unit in wandering_element(dims)]
                ])
            err = la.norm(approx_gradbf_v-gradbf_v, np.Inf)
            assert err < err_factor*h


# You can test individual routines by typing
# $ python test_modes.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
