__copyright__ = "Copyright (C) 2024 University of Illinois Board of Trustees"

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


import numpy.linalg as la
import pytest

import modepy as mp


@pytest.mark.parametrize("shape", [
                         mp.Simplex(1),
                         mp.Simplex(2),
                         mp.Simplex(3),
                         mp.Hypercube(2),
                         mp.Hypercube(3),
                         ])
@pytest.mark.parametrize("order", [0, 1, 2, 4])
@pytest.mark.parametrize("nodes_on_bdry", [False, True])
def test_nodal_mass_matrix_against_quad(
            shape: mp.Shape, order: int, nodes_on_bdry: bool,
        ) -> None:
    space = mp.space_for_shape(shape, order)
    quad_space = mp.space_for_shape(shape, 2*order)
    quad = mp.quadrature_for_space(quad_space, shape)

    if isinstance(shape, mp.Hypercube) or shape == mp.Simplex(1):
        if nodes_on_bdry:
            nodes = mp.legendre_gauss_lobatto_tensor_product_nodes(
                shape.dim, order,
            )
        else:
            nodes = mp.legendre_gauss_tensor_product_nodes(shape.dim, order)
    elif isinstance(shape, mp.Simplex):
        if nodes_on_bdry:
            nodes = mp.warp_and_blend_nodes(shape.dim, order)
        else:
            nodes = mp.VioreanuRokhlinSimplexQuadrature(order, shape.dim).nodes

    else:
        raise AssertionError()

    basis = mp.orthonormal_basis_for_space(space, shape)

    quad_mass_mat = mp.nodal_quad_mass_matrix(quad, basis.functions, nodes)
    vdm_mass_mat = mp.mass_matrix(basis, nodes)

    nodes_to_quad = mp.resampling_matrix(basis.functions, quad.nodes, nodes)

    err = la.norm(quad_mass_mat@nodes_to_quad - vdm_mass_mat)/la.norm(vdm_mass_mat)
    assert err < 1e-14


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
