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

import logging
from functools import partial

import numpy as np
import numpy.linalg as la
import pytest

import modepy as mp


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
    space = mp.PN(dims, n)
    nodes = mp.warp_and_blend_nodes(dims, n)
    basis = mp.orthonormal_basis_for_space(space, mp.Simplex(dims))
    vdm = mp.vandermonde(basis.functions, nodes)

    f = test_func(nodes[0])
    coeffs = la.solve(vdm, f)

    if False:
        from modepy.tools import plot_element_values
        plot_element_values(n, nodes, f, resample_n=70, show_nodes=True)

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
        basis = mp.orthonormal_basis_for_space(
                mp.PN(dims, inner_n), mp.Simplex(dims))
        vdm = mp.vandermonde(basis.functions, nodes)

        f = test_func(nodes[0])
        coeffs = la.solve(vdm, f)

        from modepy.modal_decay import estimate_relative_expansion_residual
        return estimate_relative_expansion_residual(
                coeffs.reshape(1, -1), dims, inner_n)

    resid = estimate_resid(n)[0]
    resid2 = estimate_resid(2*n)[0]
    print(f"{case_name}: {float(resid):g} -> {float(resid2):g}")
    assert resid2 < resid

# }}}


# {{{ test_resampling_matrix

@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("shape_cls", [mp.Simplex, mp.Hypercube])
def test_resampling_matrix(dims, shape_cls, ncoarse=5, nfine=10):
    shape = shape_cls(dims)

    coarse_space = mp.space_for_shape(shape, ncoarse)
    fine_space = mp.space_for_shape(shape, nfine)

    coarse_nodes = mp.edge_clustered_nodes_for_space(coarse_space, shape)
    coarse_basis = mp.basis_for_space(coarse_space, shape)

    fine_nodes = mp.edge_clustered_nodes_for_space(fine_space, shape)
    fine_basis = mp.basis_for_space(fine_space, shape)

    my_eye = np.dot(
            mp.resampling_matrix(fine_basis.functions, coarse_nodes, fine_nodes),
            mp.resampling_matrix(coarse_basis.functions, fine_nodes, coarse_nodes))

    assert la.norm(my_eye - np.eye(len(my_eye))) < 3e-13

    my_eye_least_squares = np.dot(
            mp.resampling_matrix(coarse_basis.functions, coarse_nodes, fine_nodes,
                least_squares_ok=True),
            mp.resampling_matrix(coarse_basis.functions, fine_nodes, coarse_nodes),
            )

    assert la.norm(my_eye_least_squares - np.eye(len(my_eye_least_squares))) < 4e-13


@pytest.mark.parametrize("space_nh", [
    mp.TensorProductSpace((mp.QN(1, 3),)),
    mp.TensorProductSpace((mp.QN(1, 3), mp.QN(1, 5))),
    mp.TensorProductSpace((mp.QN(1, 3), mp.QN(1, 5), mp.QN(1, 2))),
    mp.TensorProductSpace((mp.QN(2, 3), mp.QN(1, 2))),
    ])
def test_non_homogeneous_tensor_product_resampling(space_nh):
    shape = mp.Hypercube(space_nh.spatial_dim)
    orders_h = 5

    # {{{ construct spaces

    space_h = mp.space_for_shape(shape, orders_h)
    nodes_h = mp.equispaced_nodes_for_space(space_h, shape)
    basis_h = mp.orthonormal_basis_for_space(space_h, shape).functions

    assert nodes_h.shape[-1] == len(basis_h)
    assert len(basis_h) == space_h.space_dim

    nodes_nh = mp.equispaced_nodes_for_space(space_nh, shape)
    basis_nh = mp.orthonormal_basis_for_space(space_nh, shape).functions

    assert nodes_nh.shape[-1] == len(basis_nh)
    assert len(basis_nh) == space_nh.space_dim

    # }}}

    # {{{ check resampling

    from_h_mat = mp.resampling_matrix(basis_h, nodes_nh, nodes_h)
    to_h_mat = mp.resampling_matrix(basis_nh, nodes_h, nodes_nh)
    logger.info("cond from %.5e to %.5e", la.cond(from_h_mat), la.cond(to_h_mat))

    error = la.norm(from_h_mat @ to_h_mat - np.eye(to_h_mat.shape[1]))
    assert error < 1.0e-13, error

    nodes_0_resampled = to_h_mat @ nodes_nh[0]
    logger.info("nh_to_h error %.5e", la.norm(nodes_0_resampled - nodes_h[0]))

    nodes_1_resampled = from_h_mat @ nodes_h[0]
    logger.info("h_to_nh error: %.5e", la.norm(nodes_1_resampled - nodes_nh[0]))

    # }}}

# }}}


# {{{ test_diff_matrix

def _test_diff_matrix(space, shape, rtol=2.0e-4):
    nodes = mp.edge_clustered_nodes_for_space(space, shape)
    basis = mp.basis_for_space(space, shape)

    diff_mat = mp.differentiation_matrices(basis.functions, basis.gradients, nodes)
    if not isinstance(diff_mat, tuple):
        diff_mat = (diff_mat,)

    f = sum(np.sin(nodes[i]) for i in range(shape.dim))
    df_dx = tuple([np.cos(nodes[i]) for i in range(shape.dim)])
    df_dx_num = tuple([np.dot(diff_mat[i], f) for i in range(shape.dim)])

    for i in range(shape.dim):
        error = la.norm(df_dx[i] - df_dx_num[i]) / la.norm(df_dx[i])
        logger.info("error: %.5e", error)
        print(error)

        assert error < rtol, error


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize(("shape_cls", "order"), [
    (mp.Simplex, 5),
    (mp.Hypercube, 5),
    (mp.Hypercube, (6, 7, 5))])
def test_diff_matrix(dims, shape_cls, order):
    if isinstance(order, tuple):
        if dims == 1:
            order = order[dims]
        else:
            order = order[:dims]

    shape = shape_cls(dims)
    space = mp.space_for_shape(shape, order)
    _test_diff_matrix(space, shape)


def test_nonhomogeneous_tensor_product_diff_matrix():
    shape = mp.Hypercube(3)
    space = mp.TensorProductSpace((mp.QN(2, 5), mp.QN(1, 5)))

    _test_diff_matrix(space, shape)


@pytest.mark.parametrize("dims", [2, 3])
def test_diff_matrix_permutation(dims):
    order = 5
    space = mp.PN(dims, order)

    from pytools import (
        generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
    )
    node_tuples = list(gnitstam(order, dims))

    simplex_onb = mp.orthonormal_basis_for_space(space, mp.Simplex(dims))
    nodes = np.array(mp.warp_and_blend_nodes(dims, order, node_tuples=node_tuples))
    diff_matrices = mp.differentiation_matrices(
            simplex_onb.functions, simplex_onb.gradients, nodes)

    for iref_axis in range(dims):
        perm = mp.diff_matrix_permutation(node_tuples, iref_axis)

        assert la.norm(
                diff_matrices[iref_axis]
                - diff_matrices[0][perm][:, perm]) < 1e-10


# }}}


# {{{ face mass matrices

@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("shape_cls", [mp.Simplex, mp.Hypercube])
def test_modal_mass_matrix_for_face(dims, shape_cls, order=3):
    vol_shape = shape_cls(dims)
    vol_space = mp.space_for_shape(vol_shape, order)
    vol_basis = mp.basis_for_space(vol_space, vol_shape)

    from modepy.matrices import modal_mass_matrix_for_face
    for face in mp.faces_for_shape(vol_shape):
        face_space = mp.space_for_shape(face, order)
        face_basis = mp.basis_for_space(face_space, face)
        face_quad = mp.quadrature_for_space(mp.space_for_shape(face, 2*order), face)
        face_quad2 = mp.quadrature_for_space(
                mp.space_for_shape(face, 2*order+2), face)
        fmm = modal_mass_matrix_for_face(
                face, face_quad, face_basis.functions, vol_basis.functions)
        fmm2 = modal_mass_matrix_for_face(
                face, face_quad2, face_basis.functions, vol_basis.functions)

        error = la.norm(fmm - fmm2, np.inf) / la.norm(fmm2, np.inf)
        logger.info("fmm error: %.5e", error)
        assert error < 1e-11, f"error {error:.5e} on face {face.face_index}"

        fmm[np.abs(fmm) < 1e-13] = 0
        nnz = np.sum(fmm > 0)

        logger.info("fmm: nnz %d\n%s", nnz, fmm)


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("shape_cls", [mp.Simplex, mp.Hypercube])
def test_nodal_mass_matrix_for_face(dims, shape_cls, order=3):
    vol_shape = shape_cls(dims)
    vol_space = mp.space_for_shape(vol_shape, order)

    volume_nodes = mp.edge_clustered_nodes_for_space(vol_space, vol_shape)
    volume_basis = mp.basis_for_space(vol_space, vol_shape)

    from modepy.matrices import (
        nodal_mass_matrix_for_face,
        nodal_quad_mass_matrix_for_face,
    )
    for face in mp.faces_for_shape(vol_shape):
        face_space = mp.space_for_shape(face, order)
        face_basis = mp.basis_for_space(face_space, face)
        face_nodes = mp.edge_clustered_nodes_for_space(face_space, face)
        face_quad = mp.quadrature_for_space(mp.space_for_shape(face, 2*order), face)
        face_quad2 = mp.quadrature_for_space(
                mp.space_for_shape(face, 2*order+2), face)
        fmm = nodal_mass_matrix_for_face(
                face, face_quad, face_basis.functions, volume_basis.functions,
                volume_nodes, face_nodes)
        fmm2 = nodal_quad_mass_matrix_for_face(
                face, face_quad2, volume_basis.functions, volume_nodes)

        for f_face in face_basis.functions:
            fval_nodal = f_face(face_nodes)
            fval_quad = f_face(face_quad2.nodes)
            assert (
                    la.norm(fmm@fval_nodal - fmm2@fval_quad, np.inf)
                    / la.norm(fval_quad)) < 3e-15

        fmm[np.abs(fmm) < 1e-13] = 0
        nnz = np.sum(fmm > 0)

        logger.info("fmm: nnz %d\n%s", nnz, fmm)

        logger.info("mass matrix:\n%s",
                mp.mass_matrix(face_basis.functions, face_nodes))

# }}}


# {{{ test_estimate_lebesgue_constant

@pytest.mark.parametrize("dims", [1, 2])
@pytest.mark.parametrize("order", [3, 5, 8])
@pytest.mark.parametrize("shape_cls", [mp.Simplex, mp.Hypercube])
def test_estimate_lebesgue_constant(dims, order, shape_cls, visualize=False):
    shape = shape_cls(dims)
    space = mp.space_for_shape(shape, order)

    nodes = mp.edge_clustered_nodes_for_space(space, shape)

    from modepy.tools import estimate_lebesgue_constant
    lebesgue_constant = estimate_lebesgue_constant(order, nodes, shape=shape)
    logger.info("%s-%d/%s: %.5e", shape, dims, order, lebesgue_constant)

    if not visualize:
        return

    from modepy.tools import _evaluate_lebesgue_function
    lebesgue, _equi_node_tuples, equi_nodes = \
            _evaluate_lebesgue_function(order, nodes, shape)

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

    shape_name = shape_cls.__name__.lower()
    fig.savefig(f"estimate_lebesgue_constant_{shape_name}_{dims}_order_{order}")

# }}}


# {{{ test_hypercube_submesh

@pytest.mark.parametrize("dims", [2, 3, 4])
def test_hypercube_submesh(dims, order=3):
    shape = mp.Hypercube(dims)
    space = mp.space_for_shape(shape, order)

    node_tuples = mp.node_tuples_for_space(space)
    for i, nt in enumerate(node_tuples):
        logger.info("[%4d] nodes %s", i, nt)

    assert len(node_tuples) == (order + 1)**dims

    elements = mp.submesh_for_shape(shape, node_tuples)

    for e in elements:
        logger.info("element: %s", e)

    assert len(elements) == order**dims

# }}}


# {{{{ test_normals

@pytest.mark.parametrize("shape", [
    mp.Simplex(1),
    mp.Simplex(2),
    mp.Simplex(3),
    mp.Hypercube(1),
    mp.Hypercube(2),
    mp.Hypercube(3),
    ])
def test_normals(shape):
    vol_vertices = mp.unit_vertices_for_shape(shape)
    vol_centroid = np.mean(vol_vertices, axis=1)

    for face in mp.faces_for_shape(shape):
        face_vertices = vol_vertices[:, face.volume_vertex_indices]
        face_centroid = np.mean(face_vertices, axis=1)
        normal = mp.face_normal(face)

        assert normal @ (face_centroid-vol_centroid) > 0

        for i in range(len(face_vertices)-1):
            assert abs(
                (face_vertices[:, i+1] - face_vertices[:, 0]) @ normal) < 1e-13

        assert abs(la.norm(normal, 2) - 1) < 1e-13

# }}}


# {{{ test_tensor_product_shapes

def test_tensor_product_shapes():
    # shape, dim, nvertices, nfaces
    shapes = [
            (mp.Hypercube(1), 1, 2, 2),
            (mp.Hypercube(2), 2, 4, 4),
            (mp.Hypercube(3), 3, 8, 6),
            (mp.TensorProductShape((mp.Simplex(1),) * 2), 2, 4, 4),
            (mp.TensorProductShape((mp.Simplex(1),) * 3), 3, 8, 6),
            (mp.TensorProductShape((mp.Hypercube(1), mp.Hypercube(2))), 3, 8, 6),
            (mp.TensorProductShape((mp.Simplex(2), mp.Simplex(1))), 3, 6, 5),
            (mp.TensorProductShape((mp.Simplex(1), mp.Simplex(2))), 3, 6, 5),
            ]

    assert isinstance(mp.Hypercube(1), mp.Simplex)
    assert isinstance(mp.TensorProductShape((mp.Simplex(2),)), mp.Simplex)

    for shape, dim, nvertices, nfaces in shapes:
        assert shape.dim == dim
        assert shape.nvertices == nvertices
        assert shape.nfaces == nfaces

    with pytest.raises(NotImplementedError):
        mp.TensorProductShape((mp.Simplex(2),) * 2)

    mp.faces_for_shape(mp.Hypercube(3))

# }}}


# {{{ tensor product reshape

@pytest.mark.parametrize("dim", [2, 3])
def test_tensor_product_reshape(dim):
    interval = mp.Simplex(1)
    shape = mp.TensorProductShape((interval,) * dim)
    space = mp.TensorProductSpace(tuple([mp.PN(1, 4+i) for i in range(dim)]))
    basis = mp.basis_for_space(space, shape)

    from modepy.tools import reshape_array_for_tensor_product_space

    tol = 1e-13

    for nodes_func in [
            mp.edge_clustered_nodes_for_space,
            mp.equispaced_nodes_for_space,
            ]:
        np.set_printoptions(linewidth=200)

        nodes = nodes_func(space, shape)
        nodes_r = reshape_array_for_tensor_product_space(space, nodes)

        assert nodes_r.shape[1:] == tuple(5+i for i in range(dim))

        vdm = mp.vandermonde(basis.functions, nodes)
        _, sing_values, _ = la.svd(vdm)
        assert min(sing_values) > 1e-10

        for i in range(dim):
            # ensure that node_ids are valid indices into the nodal array
            for nid in mp.node_tuples_for_space(space):
                nodes_r[0][nid]

            for j in range(dim):
                varies_along_axis_j = np.diff(nodes_r[i], axis=j).any()
                assert varies_along_axis_j == (j == i)

            f_ax = np.exp(nodes[i])
            f_ax_modal_r = reshape_array_for_tensor_product_space(space,
                    la.solve(vdm, f_ax))

            # ensure that mode_ids are valid indices into the modal array
            for mid in basis.mode_ids:
                f_ax_modal_r[mid]

            for j in range(dim):
                # constant <=> all but the first coefficient is zero
                constant_along_axis_j = np.max(np.abs(
                    f_ax_modal_r[(slice(None),) * (j) + (slice(1, None),)]
                    )) < tol

                assert constant_along_axis_j == (j != i)

# }}}


# {{{ test_tensor_product_vdm_dim_by_dim

@pytest.mark.parametrize("dim", [2, 3])
def test_tensor_product_vdm_dim_by_dim(dim):
    """Apply tensor product Vandermonde one dimension at a time, check that the
    result matches what's obtained via the whole-space Vandermonde.
    """

    interval = mp.Simplex(1)
    shape = mp.TensorProductShape((interval,) * dim)
    space = mp.TensorProductSpace(tuple([mp.PN(1, 4+i) for i in range(dim)]))
    basis = mp.basis_for_space(space, shape)

    nodes = mp.edge_clustered_nodes_for_space(space, shape)
    vdm = mp.vandermonde(basis.functions, nodes)

    x = np.random.randn(space.space_dim)
    vdm_x = vdm @ x

    from modepy.tools import (
        reshape_array_for_tensor_product_space,
        unreshape_array_for_tensor_product_space,
    )
    x_r = reshape_array_for_tensor_product_space(space, x)
    vdm_dimbydim_x_r = x_r

    for i, (subspace, subshape) in enumerate(zip(space.bases, shape.bases)):
        subnodes = mp.edge_clustered_nodes_for_space(subspace, subshape)
        subbasis = mp.basis_for_space(subspace, subshape)
        subvdm = mp.vandermonde(subbasis.functions, subnodes)

        pre_dims = "kl"[:i]
        post_dims = "mn"[:dim-i-1]
        vdm_dimbydim_x_r = np.einsum(
                f"ij,{pre_dims}j{post_dims}->{pre_dims}i{post_dims}",
                subvdm, vdm_dimbydim_x_r)

    vdm_dimbydim_x = unreshape_array_for_tensor_product_space(
            space, vdm_dimbydim_x_r)

    err = la.norm(vdm_x - vdm_dimbydim_x)
    assert err < 1e-12, err

# }}}


# {{{ test_shape_pickling

@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("shape_cls", [mp.Simplex, mp.Hypercube])
def test_shape_pickling(dims, shape_cls):
    import pickle

    shape = shape_cls(dims)
    tp_space = mp.space_for_shape(shape, 1)

    pkl = pickle.dumps(shape)
    reloaded_shape = pickle.loads(pkl)
    assert shape == reloaded_shape

    pkl2 = pickle.dumps(tp_space)
    reloaded_tp_space = pickle.loads(pkl2)
    assert tp_space.spatial_dim == reloaded_tp_space.spatial_dim
    assert tp_space.space_dim == reloaded_tp_space.space_dim

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
