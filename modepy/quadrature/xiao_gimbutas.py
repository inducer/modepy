from __future__ import division

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
    """A (nearly) Gaussian simplicial quadrature with very few quadrature nodes,
    available for low-to-moderate orders.

    Raises :exc:`modepy.QuadratureRuleUnavailable` if no quadrature rule for the
    requested parameters is available.

    The integration domain is the unit simplex. (see :ref:`tri-coords`
    and :ref:`tet-coords`)

    Inherits from :class:`modepy.Quadrature`. See there for the interface
    to obtain nodes and weights.

    .. attribute:: exact_to

        The total degree up to which the quadrature is exact.

    See

        H. Xiao and Z. Gimbutas, "A numerical algorithm for the construction of
        efficient quadrature rules in two and higher dimensions," Computers &
        Mathematics with Applications, vol. 59, no. 2, pp. 663-676, 2010.
        http://dx.doi.org/10.1016/j.camwa.2009.10.027
    """

    def __init__(self, order, dims):
        """
        :arg order: The total degree to which the quadrature rule is exact.
        :arg dims: The number of dimensions for the quadrature rule.
            2 for quadrature on triangles and 3 for tetrahedra.
        """

        if dims == 2:
            from modepy.quadrature.xg_quad_data import triangle_table as table
        elif dims == 3:
            from modepy.quadrature.xg_quad_data import tetrahedron_table as table
        else:
            raise ValueError("invalid dimensionality")
        try:
            order_table = table[order]
        except KeyError:
            raise QuadratureRuleUnavailable

        from modepy.tools import EQUILATERAL_TO_UNIT_MAP
        e2u = EQUILATERAL_TO_UNIT_MAP[dims]

        nodes = e2u(order_table["points"].T)
        wts = order_table["weights"]*e2u.jacobian

        Quadrature.__init__(self, nodes, wts)

        self.exact_to = order
