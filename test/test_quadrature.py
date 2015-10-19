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

from six.moves import range

import numpy as np
import pytest
import modepy as mp


def test_transformed_quadrature():
    """Test 1D quadrature on arbitrary intervals"""

    def gaussian_density(x, mu, sigma):
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

    from modepy.quadrature import Transformed1DQuadrature
    from modepy.quadrature.jacobi_gauss import LegendreGaussQuadrature

    mu = 17
    sigma = 12
    tq = Transformed1DQuadrature(LegendreGaussQuadrature(20), mu-6*sigma, mu+6*sigma)

    result = tq(lambda x: gaussian_density(x, mu, sigma))
    assert abs(result - 1) < 1e-9


def test_gauss_quadrature():
    from modepy.quadrature.jacobi_gauss import LegendreGaussQuadrature

    for s in range(9+1):
        cub = LegendreGaussQuadrature(s)
        for deg in range(2*s+1 + 1):
            def f(x):
                return x**deg
            i_f = cub(f)
            i_f_true = 1/(deg+1)*(1-(-1)**(deg+1))
            err = abs(i_f - i_f_true)
            assert err < 2e-15


@pytest.mark.parametrize(("quad_class", "highest_order"), [
    (mp.XiaoGimbutasSimplexQuadrature, None),
    (mp.VioreanuRokhlinSimplexQuadrature, None),
    (mp.GrundmannMoellerSimplexQuadrature, 3),
    ])
@pytest.mark.parametrize("dim", [2, 3])
def test_simplex_quadrature(quad_class, highest_order, dim):
    """Check that quadratures on simplices works as advertised"""
    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
            as gnitstam
    from modepy.tools import Monomial

    order = 1
    while True:
        try:
            quad = quad_class(order, dim)
        except mp.QuadratureRuleUnavailable:
            print(("UNAVAIL", quad_class, order))
            break

        if isinstance(quad_class, mp.VioreanuRokhlinSimplexQuadrature):
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
            assert err < 6e-15, (err, comb, i_f, ref)

        order += 1
        if highest_order is not None and order >= highest_order:
            break


# You can test individual routines by typing
# $ python test_quadrature.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
