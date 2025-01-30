from __future__ import annotations


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


from typing import cast

import numpy as np
import numpy.linalg as la
import pytest

import modepy as mp


@pytest.mark.parametrize("shape", [
                         mp.Hypercube(1),
                         mp.Hypercube(2),
                         mp.Hypercube(3),
                         ])
def test_tensor_product_diag_mass_matrix(shape: mp.Shape) -> None:
    shape = mp.Simplex(1)

    for order in range(16):
        space = mp.space_for_shape(shape, order)
        basis = mp.orthonormal_basis_for_space(space, shape)

        gl_quad = mp.LegendreGaussTensorProductQuadrature(order, shape.dim)
        gl_ref_mass_mat = mp.mass_matrix(basis, gl_quad.nodes)
        gl_diag_mass_mat = np.diag(mp.spectral_diag_nodal_mass_matrix(gl_quad))

        gl_err = (
            la.norm(gl_ref_mass_mat - gl_diag_mass_mat, "fro")
            / la.norm(gl_ref_mass_mat, "fro")
        )

        assert gl_err < 1e-14

        if order == 0:
            # no single-node Lobatto quadratures
            continue

        gll_quad = mp.LegendreGaussLobattoTensorProductQuadrature(order, shape.dim)
        gll_ref_mass_mat = mp.mass_matrix(basis, gll_quad.nodes)
        gll_diag_mass_mat = np.diag(mp.spectral_diag_nodal_mass_matrix(gll_quad))

        # Note that gll_diag_mass_mat is not a good approximation of gll_ref_mass_mat
        # in the matrix norm sense!

        for mid, func in zip(basis.mode_ids, basis.functions, strict=True):
            if max(cast(tuple[int, ...], mid)) < order - 1:
                err = np.abs(
                    gll_ref_mass_mat @ func(gll_quad.nodes)
                    - gll_diag_mass_mat @ func(gll_quad.nodes))
                assert np.max(err) < 1e-14


@pytest.mark.parametrize("shape_cls", [mp.Hypercube, mp.Simplex])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [0, 1, 2, 4])
@pytest.mark.parametrize("nodes_on_bdry", [False, True])
@pytest.mark.parametrize("test_weak_d_dr", [False, True])
def test_bilinear_forms(
            shape_cls: type[mp.Shape],
            dim: int,
            order: int,
            nodes_on_bdry: bool,
            test_weak_d_dr: bool
        ) -> None:
    shape = shape_cls(dim)
    space = mp.space_for_shape(shape, order)

    quad_space = mp.space_for_shape(shape, 2*order)
    quad = mp.quadrature_for_space(quad_space, shape)

    if nodes_on_bdry:
        nodes = mp.edge_clustered_nodes_for_space(space, shape)
    else:
        if isinstance(shape, mp.Hypercube) or shape == mp.Simplex(1):
            nodes = mp.legendre_gauss_tensor_product_nodes(shape.dim, order)
        elif isinstance(shape, mp.Simplex):
            nodes = mp.VioreanuRokhlinSimplexQuadrature(order, shape.dim).nodes
        else:
            raise AssertionError()

    basis = mp.orthonormal_basis_for_space(space, shape)

    if test_weak_d_dr and order not in [0, 1]:
        mass_inv = mp.inverse_mass_matrix(basis, nodes)

        for ax in range(dim):
            f = 1 - nodes[ax]**2
            fp = -2*nodes[ax]

            weak_operator = mp.nodal_quadrature_bilinear_form_matrix(
                quadrature=quad,
                test_basis_functions=basis.functions,
                trial_basis_functions=basis.functions,
                input_nodes=nodes,
                test_derivatives=basis.derivatives(ax)
            )

            err = la.norm(mass_inv @ weak_operator.T @ f - fp) / la.norm(fp)
            assert err <= 1e-12
    else:
        quad_mass_mat = mp.nodal_quadrature_bilinear_form_matrix(
            quadrature=quad,
            test_basis_functions=basis.functions,
            trial_basis_functions=basis.functions,
            input_nodes=nodes
        )

        vdm_mass_mat = mp.mass_matrix(basis, nodes)
        err = la.norm(quad_mass_mat - vdm_mass_mat) / la.norm(vdm_mass_mat)
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
