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
import numpy.linalg as la
import pytest
import modepy.shapes as shp
import modepy.nodes as nd


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_barycentric_coordinate_map(dims):
    n = 100
    from modepy.tools import (
            unit_to_barycentric,
            barycentric_to_unit,
            barycentric_to_equilateral,
            equilateral_to_unit,)

    rng = np.random.Generator(np.random.PCG64(17))
    unit = nd.random_nodes_for_shape(shp.Simplex(dims), n, rng=rng)

    bary = unit_to_barycentric(unit)
    assert (np.abs(np.sum(bary, axis=0) - 1) < 1e-15).all()
    assert (bary >= 0).all()
    unit2 = barycentric_to_unit(bary)
    assert la.norm(unit-unit2) < 1e-14

    equi = barycentric_to_equilateral(bary)
    unit3 = equilateral_to_unit(equi)
    assert la.norm(unit-unit3) < 1e-14


def test_warp():
    """Check some assumptions on the node warp factor calculator"""
    n = 17
    from functools import partial

    def wfc(x):
        return nd.warp_factor(n, np.array([x]), scaled=False)[0]

    assert abs(wfc(-1)) < 1e-12
    assert abs(wfc(1)) < 1e-12

    from modepy.quadrature.jacobi_gauss import LegendreGaussQuadrature

    lgq = LegendreGaussQuadrature(n)
    assert abs(lgq(partial(nd.warp_factor, n, scaled=False))) < 6e-14


def test_tri_face_node_distribution():
    """Test whether the nodes on the faces of the triangle are distributed
    according to the same proportions on each face.

    If this is not the case, then reusing the same face mass matrix
    for each face would be invalid.
    """

    n = 8
    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
            as gnitstam
    node_tuples = list(gnitstam(n, 2))

    unodes = nd.warp_and_blend_nodes(2, n, node_tuples)

    faces = [
            [i for i, nt in enumerate(node_tuples) if nt[0] == 0],
            [i for i, nt in enumerate(node_tuples) if nt[1] == 0],
            [i for i, nt in enumerate(node_tuples) if sum(nt) == n]
            ]

    projected_face_points = []
    for face_i in faces:
        start = unodes[:, face_i[0]]
        end = unodes[:, face_i[-1]]
        direction = end-start
        direction /= np.dot(direction, direction)
        pfp = np.array([np.dot(direction, unodes[:, i]-start) for i in face_i])
        projected_face_points.append(pfp)

    first_points = projected_face_points[0]
    for points in projected_face_points[1:]:
        error = la.norm(points-first_points, np.Inf)
        assert error < 1e-15


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("n", [1, 3, 6])
def test_simp_nodes(dims, n):
    """Verify basic assumptions on simplex interpolation nodes"""

    eps = 1e-10

    unodes = nd.warp_and_blend_nodes(dims, n)
    assert (unodes >= -1-eps).all()
    assert (np.sum(unodes) <= eps).all()


def test_affine_map():
    """Check that our cheapo geometry-targeted linear algebra actually works."""
    from modepy.tools import AffineMap
    # for d in [3]:
    for d in range(1, 5):
        for _i in range(100):
            a = np.random.randn(d, d)+10*np.eye(d)
            b = np.random.randn(d)

            m = AffineMap(a, b)
            x = np.random.randn(d, 10)

            assert la.norm(x-m.inverse(m(x))) < 1e-10


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_tensor_product_nodes(dim):
    nnodes = 10
    nodes_1d = np.arange(nnodes)
    nodes = nd.tensor_product_nodes(dim, nodes_1d)

    assert np.allclose(
            nodes[0],
            np.array(nodes_1d.tolist() * nnodes**(dim - 1)))


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_nonhomogeneous_tensor_product_nodes(dim):
    nnodes = (3, 7, 5, 4)[:dim]
    nodes = nd.tensor_product_nodes([
        np.arange(n) for n in nnodes
        ])

    assert np.allclose(
            nodes[0],
            list(range(nnodes[-1])) * int(np.prod(nnodes[:-1]))
            )


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("shape_cls", [shp.Hypercube, shp.Simplex])
def test_order0_nodes(dim, shape_cls):
    shape = shape_cls(dim)
    import modepy as mp
    space = mp.space_for_shape(shape, order=0)

    centroid = (np.mean(mp.unit_vertices_for_shape(shape), axis=1)
            .reshape(-1, 1))
    nodes = mp.equispaced_nodes_for_space(space, shape)
    assert not np.isnan(nodes).any()
    assert np.allclose(centroid, nodes)

    nodes = mp.edge_clustered_nodes_for_space(space, shape)
    assert not np.isnan(nodes).any()
    assert np.allclose(centroid, nodes)


# You can test individual routines by typing
# $ python test_nodes.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
