__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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
import modepy as mp

from functools import partial
import pytest

import logging
logger = logging.getLogger(__name__)


# {{{ modal decay test functions

def jump(where, x):
    result = np.empty_like(x)
    result[x >= where] = 1
    result[x < where] = 0
    return result


def smooth_jump(where, x):
    return np.arctan(100*(x-where))/(0.5*np.pi)


def kink(where, x):
    result = np.empty_like(x)
    result[x >= where] = x[x >= where]
    result[x < where] = 0
    return result


def c1(where, x):
    result = np.empty_like(x)
    result[x >= where] = x[x >= where]**2
    result[x < where] = 0
    return result


def sample_polynomial(x):
    return 3*x**2 + 5*x + 3


def constant(x):
    return 5+0*x

# }}}


# {{{ test_modal_decay

@pytest.mark.parametrize(("case_name", "test_func", "dims", "n", "expected_expn"), [
    ("jump-1d", partial(jump, 0), 1, 10, -1),
    ("kink-1d", partial(kink, 0), 1, 10, -1.7),  # Slightly off from -2 (same in paper)  # noqa
    ("c1-1d", partial(c1, 0), 1, 10, -3),

    # Offset 1D tests
    ("offset-jump-1d", partial(jump, 0.8), 1, 20, -0.7),
    ("offset-kink-1d", partial(kink, 0.8), 1, 20, -0.7),  # boo
    ("offset-c1-1d", partial(c1, 0.8), 1, 20, -0.8),  # boo

    # 2D tests
    # A harsh jump introduces high-frequency wiggles transverse to the jump edge.
    # Therefore, use a smooth one.
    ("jump-2d", partial(smooth_jump, -0.1), 2, 15, -1.1),
    ("kink-2d", partial(kink, 0), 2, 15, -1.6),
    ("c1-2d", partial(c1, -0.1), 2, 15, -2.3),
    ])
def test_modal_decay(case_name, test_func, dims, n, expected_expn):
    nodes = mp.warp_and_blend_nodes(dims, n)
    basis = mp.simplex_onb(dims, n)
    vdm = mp.vandermonde(basis, nodes)

    f = test_func(nodes[0])
    coeffs = la.solve(vdm, f)

    if 0:
        from modepy.tools import plot_element_values
        plot_element_values(n, nodes, f, resample_n=70,
                show_nodes=True)

    from modepy.modal_decay import fit_modal_decay
    expn, _ = fit_modal_decay(coeffs.reshape(1, -1), dims, n)
    expn = expn[0]

    print(f"{case_name}: computed: {expn:g}, expected: {expected_expn:g}")
    assert abs(expn-expected_expn) < 0.1


@pytest.mark.parametrize(("case_name", "test_func", "dims", "n"), [
    ("sin-1d", np.sin, 1, 5),
    ("poly-1d", sample_polynomial, 1, 5),
    ("const-1d", constant, 1, 5),

    ("sin-2d", np.sin, 2, 5),
    ("poly-2d", sample_polynomial, 2, 5),
    ("const-2d", constant, 2, 5),
    ])
def test_residual_estimation(case_name, test_func, dims, n):
    def estimate_resid(inner_n):
        nodes = mp.warp_and_blend_nodes(dims, inner_n)
        basis = mp.simplex_onb(dims, inner_n)
        vdm = mp.vandermonde(basis, nodes)

        f = test_func(nodes[0])
        coeffs = la.solve(vdm, f)

        from modepy.modal_decay import estimate_relative_expansion_residual
        return estimate_relative_expansion_residual(
                coeffs.reshape(1, -1), dims, inner_n)

    resid = estimate_resid(n)
    resid2 = estimate_resid(2*n)
    print(f"{case_name}: {float(resid):g} -> {float(resid2):g}")
    assert resid2 < resid

# }}}


# {{{ test_resampling_matrix

@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("eltype", ["simplex", "tensor"])
def test_resampling_matrix(dims, eltype):
    ncoarse = 5
    nfine = 10

    if eltype == "simplex":
        coarse_nodes = mp.warp_and_blend_nodes(dims, ncoarse)
        fine_nodes = mp.warp_and_blend_nodes(dims, nfine)

        coarse_basis = mp.simplex_onb(dims, ncoarse)
        fine_basis = mp.simplex_onb(dims, nfine)
    elif eltype == "tensor":
        coarse_nodes = mp.tensor_product_nodes(dims,
                mp.warp_and_blend_nodes(1, ncoarse)[0])
        fine_nodes = mp.tensor_product_nodes(dims,
                mp.warp_and_blend_nodes(1, nfine)[0])

        coarse_basis = mp.tensor_product_basis(dims,
                mp.simplex_onb(1, ncoarse))
        fine_basis = mp.tensor_product_basis(dims,
                mp.simplex_onb(1, nfine))
    else:
        raise ValueError(f"unknown element type: {eltype}")

    my_eye = np.dot(
            mp.resampling_matrix(fine_basis, coarse_nodes, fine_nodes),
            mp.resampling_matrix(coarse_basis, fine_nodes, coarse_nodes))

    assert la.norm(my_eye - np.eye(len(my_eye))) < 3e-13

    my_eye_least_squares = np.dot(
            mp.resampling_matrix(coarse_basis, coarse_nodes, fine_nodes,
                least_squares_ok=True),
            mp.resampling_matrix(coarse_basis, fine_nodes, coarse_nodes),
            )

    assert la.norm(my_eye_least_squares - np.eye(len(my_eye_least_squares))) < 4e-13

# }}}


# {{{ test_diff_matrix

@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("eltype", ["simplex", "tensor"])
def test_diff_matrix(dims, eltype):
    n = 5

    if eltype == "simplex":
        nodes = mp.warp_and_blend_nodes(dims, n)
        basis = mp.simplex_onb(dims, n)
        grad_basis = mp.grad_simplex_onb(dims, n)
    elif eltype == "tensor":
        nodes = mp.tensor_product_nodes(dims,
                mp.warp_and_blend_nodes(1, n)[0])
        basis = mp.tensor_product_basis(dims,
                mp.simplex_onb(1, n))
        grad_basis = mp.grad_tensor_product_basis(dims,
                mp.simplex_onb(1, n), mp.grad_simplex_onb(1, n))
    else:
        raise ValueError(f"unknown element type: {eltype}")

    diff_mat = mp.differentiation_matrices(basis, grad_basis, nodes)
    if isinstance(diff_mat, tuple):
        diff_mat = diff_mat[0]

    f = np.sin(nodes[0])

    df_dx = np.cos(nodes[0])
    df_dx_num = np.dot(diff_mat, f)

    error = la.norm(df_dx - df_dx_num) / la.norm(df_dx)
    logger.info("error: %.5e", error)
    assert error < 2.0e-4, error


@pytest.mark.parametrize("dims", [2, 3])
def test_diff_matrix_permutation(dims):
    order = 5

    from pytools import \
            generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam
    node_tuples = list(gnitstam(order, dims))

    simplex_onb = mp.simplex_onb(dims, order)
    grad_simplex_onb = mp.grad_simplex_onb(dims, order)
    nodes = np.array(mp.warp_and_blend_nodes(dims, order, node_tuples=node_tuples))
    diff_matrices = mp.differentiation_matrices(simplex_onb, grad_simplex_onb, nodes)

    for iref_axis in range(dims):
        perm = mp.diff_matrix_permutation(node_tuples, iref_axis)

        assert la.norm(
                diff_matrices[iref_axis]
                - diff_matrices[0][perm][:, perm]) < 1e-10

# }}}


# {{{ test_face_mass_matrix

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_modal_face_mass_matrix(dim, order=3):
    from modepy.tools import unit_vertices
    all_verts = unit_vertices(dim).T

    basis = mp.simplex_onb(dim, order)

    # np.set_printoptions(linewidth=200)

    from modepy.matrices import modal_face_mass_matrix
    for iface in range(dim+1):
        verts = np.hstack([all_verts[:, :iface], all_verts[:, iface+1:]])

        fmm = modal_face_mass_matrix(basis, order, verts)
        fmm2 = modal_face_mass_matrix(basis, order+1, verts)

        assert la.norm(fmm-fmm2, np.inf) < 1e-11

        fmm[np.abs(fmm) < 1e-13] = 0

        print(fmm)
        nnz = np.sum(fmm > 0)
        print(nnz)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_nodal_face_mass_matrix(dim, order=3):
    from modepy.tools import unit_vertices
    all_verts = unit_vertices(dim).T

    basis = mp.simplex_onb(dim, order)

    np.set_printoptions(linewidth=200)

    from modepy.matrices import nodal_face_mass_matrix
    volume_nodes = mp.warp_and_blend_nodes(dim, order)
    face_nodes = mp.warp_and_blend_nodes(dim-1, order)

    for iface in range(dim+1):
        verts = np.hstack([all_verts[:, :iface], all_verts[:, iface+1:]])

        fmm = nodal_face_mass_matrix(basis, volume_nodes, face_nodes, order,
                verts)
        fmm2 = nodal_face_mass_matrix(basis, volume_nodes, face_nodes, order+1,
                verts)

        assert la.norm(fmm-fmm2, np.inf) < 1e-11

        fmm[np.abs(fmm) < 1e-13] = 0

        print(fmm)
        nnz = np.sum(np.abs(fmm) > 0)
        print(nnz)

    print(mp.mass_matrix(
        mp.simplex_onb(dim-1, order),
        mp.warp_and_blend_nodes(dim-1, order), ))

# }}}


# {{{ test_estimate_lebesgue_constant

@pytest.mark.parametrize("dims", [1, 2])
@pytest.mark.parametrize("order", [3, 5, 8])
@pytest.mark.parametrize("domain", ["simplex", "hypercube"])
def test_estimate_lebesgue_constant(dims, order, domain, visualize=False):
    logging.basicConfig(level=logging.INFO)

    if domain == "simplex":
        nodes = mp.warp_and_blend_nodes(dims, order)
    elif domain == "hypercube":
        from modepy.nodes import legendre_gauss_lobatto_tensor_product_nodes
        nodes = legendre_gauss_lobatto_tensor_product_nodes(dims, order)
    else:
        raise ValueError(f"unknown domain: '{domain}'")

    from modepy.tools import estimate_lebesgue_constant
    lebesgue_constant = estimate_lebesgue_constant(order, nodes, domain=domain)
    logger.info("%s-%d/%s: %.5e", domain, dims, order, lebesgue_constant)

    if not visualize:
        return

    from modepy.tools import _evaluate_lebesgue_function
    lebesgue, equi_node_tuples, equi_nodes = \
            _evaluate_lebesgue_function(order, nodes, domain)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca()
    ax.grid()

    if dims == 1:
        ax.plot(equi_nodes[0], lebesgue)
        ax.set_xlabel("$x$")
        ax.set_ylabel(fr"$\lambda_{order}$")
    elif dims == 2:
        ax.plot(nodes[0], nodes[1], "ko")
        p = ax.tricontourf(equi_nodes[0], equi_nodes[1], lebesgue, levels=16)
        fig.colorbar(p)
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_aspect("equal")
    else:
        raise ValueError(f"unsupported dimension: {dims}")

    fig.savefig(f"estimate_lebesgue_constant_{domain}_{dims}_order_{order}")

# }}}


# {{{ test_hypercube_submesh

@pytest.mark.parametrize("dims", [2, 3, 4])
def test_hypercube_submesh(dims):
    from modepy.tools import hypercube_submesh
    from pytools import generate_nonnegative_integer_tuples_below as gnitb

    node_tuples = list(gnitb(3, dims))

    for i, nt in enumerate(node_tuples):
        logger.info("[%4d] nodes %s", i, nt)

    assert len(node_tuples) == 3**dims

    elements = hypercube_submesh(node_tuples)

    for e in elements:
        logger.info("element: %s", e)

    assert len(elements) == 2**dims

# }}}


# You can test individual routines by typing
# $ python test_tools.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
