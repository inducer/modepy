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

from modepy.quadrature import Quadrature, QuadratureRuleUnavailable


class VioreanuRokhlinSimplexQuadrature(Quadrature):
    """Simplicial quadratures with symmetric node sets and positive weights
    suitable for well-conditioned interpolation.

    The integration domain is the unit simplex (see :ref:`tri-coords`
    and :ref:`tet-coords`). When using these nodes, please acknowledge Zydrunas
    Gimbutas, who generated them as follows:

    * The 2D nodes are based on the interpolation node set derived
      in the article [Vioreanu2011]_.

      Note that in Vioreanu's tables, only orders 5, 6, 9, and 12 are rotationally
      symmetric, which gives one extra order for integration and better
      interpolation conditioning. Also note that since the tables have been
      re-generated independently, the nodes and weights may be different.

    * The 3D nodes were derived from the :func:`modepy.warp_and_blend_nodes`.

    * A tightening algorithm was then applied, as described in [Vioreanu2012]_.

    .. [Vioreanu2011] B. Vioreanu and V. Rokhlin,
        *Spectra of Multiplication Operators as a Numerical Tool*,
        Yale CS Tech Report 1443.
        `PDF <http://www.cs.yale.edu/publications/techreports/tr1443.pdf>`__

    .. [Vioreanu2012] B. Vioreanu,
        *Spectra of Multiplication Operators as a Numerical Tool*,
        Yale University, 2012.
        `PDF <http://gradworks.umi.com/3525285.pdf>`__

    .. versionadded :: 2013.3

    .. automethod:: __init__
    .. automethod:: __call__
    """

    # FIXME: most other functionality in modepy uses 'dims, order' as the
    # argument order convention.
    def __init__(self, order: int, dims: int) -> None:
        """
        :arg order: The total degree to which the quadrature rule is exact
            for *interpolation*.
        :arg dims: The number of dimensions for the quadrature rule.
            2 for quadrature on triangles and 3 for tetrahedra.

        :raises: :exc:`modepy.QuadratureRuleUnavailable` if no quadrature rule
            for the requested parameters is available.
        """

        if dims == 2:
            from modepy.quadrature.vr_quad_data_tri import triangle_data as table
            ref_volume = 2.0
        elif dims == 3:
            from modepy.quadrature.vr_quad_data_tet import tetrahedron_data as table
            ref_volume = 4/3
        else:
            raise QuadratureRuleUnavailable(f"invalid dimension: '{dims}'")

        from modepy.tools import EQUILATERAL_TO_UNIT_MAP
        e2u = EQUILATERAL_TO_UNIT_MAP[dims]

        try:
            order_table = table[order]
        except KeyError:
            raise QuadratureRuleUnavailable(f"Unsupported order: {order}") from None

        nodes = e2u(order_table["points"])
        wts = order_table["weights"]
        wts = wts * (ref_volume/np.sum(wts))

        super().__init__(nodes, wts, exact_to=order_table["quad_degree"])
