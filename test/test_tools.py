from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

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

    print(("%s: computed: %g, expected: %g" % (case_name, expn, expected_expn)))
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
    print(("%s: %g -> %g" % (case_name, resid, resid2)))
    assert resid2 < resid


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_resampling_matrix(dims):
    ncoarse = 5
    nfine = 10

    coarse_nodes = mp.warp_and_blend_nodes(dims, ncoarse)
    fine_nodes = mp.warp_and_blend_nodes(dims, nfine)

    coarse_basis = mp.simplex_onb(dims, ncoarse)
    fine_basis = mp.simplex_onb(dims, nfine)

    my_eye = np.dot(
            mp.resampling_matrix(fine_basis, coarse_nodes, fine_nodes),
            mp.resampling_matrix(coarse_basis, fine_nodes, coarse_nodes))

    assert la.norm(my_eye - np.eye(len(my_eye))) < 1e-13

    my_eye_least_squares = np.dot(
            mp.resampling_matrix(coarse_basis, coarse_nodes, fine_nodes,
                least_squares_ok=True),
            mp.resampling_matrix(coarse_basis, fine_nodes, coarse_nodes),
            )

    assert la.norm(my_eye_least_squares - np.eye(len(my_eye_least_squares))) < 4e-13


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_diff_matrix(dims):
    n = 5
    nodes = mp.warp_and_blend_nodes(dims, n)

    f = np.sin(nodes[0])
    df_dx = np.cos(nodes[0])

    diff_mat = mp.differentiation_matrices(
            mp.simplex_onb(dims, n),
            mp.grad_simplex_onb(dims, n),
            nodes)
    if isinstance(diff_mat, tuple):
        diff_mat = diff_mat[0]
    df_dx_num = np.dot(diff_mat, f)

    print((la.norm(df_dx-df_dx_num)))
    assert la.norm(df_dx-df_dx_num) < 1e-3


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


# You can test individual routines by typing
# $ python test_tools.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
