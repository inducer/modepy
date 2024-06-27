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


from modepy.quadrature import Quadrature, QuadratureRuleUnavailable


class XiaoGimbutasSimplexQuadrature(Quadrature):
    """A (nearly) Gaussian simplicial quadrature with very few quadrature nodes
    from [Xiao2010]_.

    This rule is available for low-to-moderate orders. The integration domain is
    the unit simplex (see :ref:`tri-coords` and :ref:`tet-coords`).

    .. [Xiao2010] H. Xiao and Z. Gimbutas,
        *A numerical algorithm for the construction of efficient quadrature rules
        in two and higher dimensions*,
        Computers & Mathematics with Applications, vol. 59, no. 2, pp. 663-676, 2010.
        `DOI <http://dx.doi.org/10.1016/j.camwa.2009.10.027>`__

    .. automethod:: __init__
    .. automethod:: __call__
    """

    # FIXME: most other functionality in modepy uses 'dims, order' as the
    # argument order convention.
    def __init__(self, order: int, dims: int) -> None:
        """
        :arg order: the total degree to which the quadrature rule is exact.
        :arg dims: the number of dimensions for the quadrature rule.
            2 for quadrature on triangles and 3 for tetrahedra.
        :raises: :exc:`modepy.QuadratureRuleUnavailable` if no quadrature rule
            for therequested parameters is available.
        """

        if dims == 2:
            from modepy.quadrature.xg_quad_data import triangle_table as table
        elif dims == 3:
            from modepy.quadrature.xg_quad_data import tetrahedron_table as table
        else:
            raise QuadratureRuleUnavailable(f"invalid dimension: '{dims}'")

        try:
            order_table = table[order]
        except KeyError:
            raise QuadratureRuleUnavailable(f"Unsupported order: {order}") from None

        from modepy.tools import EQUILATERAL_TO_UNIT_MAP
        e2u = EQUILATERAL_TO_UNIT_MAP[dims]

        nodes = e2u(order_table["points"].T)
        wts = order_table["weights"]*e2u.jacobian

        super().__init__(nodes, wts, exact_to=order)
