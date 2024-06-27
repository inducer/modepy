__copyright__ = "Copyright (C) 2020 Alexandru Fikl"

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

from modepy.quadrature import Quadrature, QuadratureRuleUnavailable


class WitherdenVincentQuadrature(Quadrature):
    """Symmetric quadrature rules with positive weights for rectangles and
    hexahedra from [Witherden2015]_.

    The integration domain is the unit hypercube :math:`[-1, 1]^d`, where :math:`d`
    is the dimension.

    .. [Witherden2015] F. D. Witherden, P. E. Vincent,
        *On the Identification of Symmetric Quadrature Rules for Finite
        Element Methods*,
        Computers & Mathematics with Applications, Vol. 69, pp. 1232--1241, 2015.
        `DOI <http://dx.doi.org/10.1016/j.camwa.2015.03.017>`__

    .. versionadded: 2020.3

    .. automethod:: __init__
    .. automethod:: __call__
    """

    # FIXME: most other functionality in modepy uses 'dims, order' as the
    # argument order convention.
    def __init__(self, order: int, dims: int) -> None:
        if dims == 2:
            from modepy.quadrature.witherden_vincent_quad_data import (
                quad_data as table,
            )
        elif dims == 3:
            from modepy.quadrature.witherden_vincent_quad_data import (
                hex_data as table,
            )
        else:
            raise QuadratureRuleUnavailable(f"invalid dimension: '{dims}'")

        try:
            rule = table[order]
        except KeyError:
            raise QuadratureRuleUnavailable(f"Unsupported order: {order}") from None

        super().__init__(rule["points"], rule["weights"],
                         exact_to=rule["quad_degree"])
