from __future__ import annotations


__copyright__ = "Copyright (C) 2024 Alexandru Fikl"

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

from modepy.quadrature import Quadrature, QuadratureRuleUnavailable


class JaskowiecSukumarQuadrature(Quadrature):
    """Symmetric quadrature rules with positive weights for tetrahedra from
    [Jaskowiec2021]_.

    The integration domain is the unit tetrahedron.

    .. [Jaskowiec2021] J. Jaśkowiec, N. Sukumar,
        *High‐order Symmetric Cubature Rules for Tetrahedra and Pyramids*,
        International Journal for Numerical Methods in Engineering,
        Vol. 122, pp. 148--171, 2021,
        `DOI <https://doi.org/10.1002/nme.6528>`__

    .. automethod:: __init__
    .. automethod:: __call__
    """

    # FIXME: most other functionality in modepy uses 'dims, order' as the
    # argument order convention.
    def __init__(self, order: int, dims: int) -> None:
        if dims == 3:
            from modepy.quadrature.jaskowiec_sukumar_tet_data import (
                tet_data as table,
            )
        else:
            raise QuadratureRuleUnavailable(f"invalid dimension: '{dims}'")

        try:
            rule = table[order]
        except KeyError:
            raise QuadratureRuleUnavailable(f"Unsupported order: {order}") from None

        points = 2 * np.array(rule["points"]) - 1
        weights = 4 / 3 * np.array(rule["weights"])
        exact_to = rule["quad_degree"]
        assert isinstance(exact_to, int)

        super().__init__(points, weights, exact_to=exact_to)
