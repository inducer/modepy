from __future__ import annotations


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

import modepy.nodes as nd
import modepy.shapes as shp


# {{{ test_barycentric_coordinate_map

@pytest.mark.parametrize("dims", [1, 2, 3])
def test_barycentric_coordinate_map(dims):
    n = 100
    from modepy.tools import (
        barycentric_to_equilateral,
        barycentric_to_unit,
        equilateral_to_unit,
        unit_to_barycentric,
    )

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

# }}}


# {{{ test_warp

def test_warp():
    """Check some assumptions on the node warp factor calculator"""
    n = 17
    from functools import partial

    def wfc(x):
        return nd.warp_factor(n, np.array([x]), scaled=False)[0]

    assert abs(wfc(-1)) < 1e-12
    assert abs(wfc(1)) < 1e-12

    from modepy.quadrature.jacobi_gauss import LegendreGaussQuadrature

    lgq = LegendreGaussQuadrature(n, force_dim_axis=True)
    assert abs(lgq(partial(nd.warp_factor, n, scaled=False))) < 6e-14

# }}}


# {{{ test_tri_face_node_distribution

def test_tri_face_node_distribution():
    """Test whether the nodes on the faces of the triangle are distributed
    according to the same proportions on each face.

    If this is not the case, then reusing the same face mass matrix
    for each face would be invalid.
    """

    n = 8
    from pytools import (
        generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
    )
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
        error = la.norm(points-first_points, np.inf)
        assert error < 1e-15

# }}}


# {{{ test_simplex_nodes

@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("n", [1, 3, 6])
def test_simplex_nodes(dims, n):
    """Verify basic assumptions on simplex interpolation nodes"""

    eps = 1e-10

    unodes = nd.warp_and_blend_nodes(dims, n)
    assert (unodes >= -1-eps).all()
    assert (np.sum(unodes) <= eps).all()

# }}}


# {{{ test_affine_map

def test_affine_map():
    """Check that our cheapo geometry-targeted linear algebra actually works."""
    from modepy.tools import AffineMap

    rng = np.random.default_rng(seed=42)

    for d in range(1, 5):
        for _i in range(100):
            a = rng.normal(size=(d, d)) + 10 * np.eye(d)
            b = rng.normal(size=d)

            m = AffineMap(a, b)
            x = rng.normal(size=(d, 10))

            assert la.norm(x-m.inverse(m(x))) < 1e-10

# }}}


# {{{ test_tensor_product_nodes

@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_tensor_product_nodes(dim):
    nnodes = 10
    nodes_1d = np.arange(nnodes)
    nodes = nd.tensor_product_nodes(dim, nodes_1d)

    assert np.allclose(
            nodes[0],
            np.array(nodes_1d.tolist() * nnodes**(dim - 1)))

# }}}


# {{{ test_nonhomogeneous_tensor_product_nodes

@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_nonhomogeneous_tensor_product_nodes(dim):
    nnodes = (3, 7, 5, 4)[:dim]
    nodes = nd.tensor_product_nodes([
        np.arange(n) for n in nnodes
        ])

    assert np.allclose(
            nodes[0],
            list(range(nnodes[0])) * int(np.prod(nnodes[1:]))
            )

# }}}


# {{{ test_order0_nodes

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

# }}}


# {{{ test_tensor_product_shape_nodes

@pytest.mark.parametrize("shape", ["square", "cube", "squared_cube", "prism"])
def test_tensor_product_shape_nodes(shape, visualize=False):
    order = (5, 3, 4)

    if shape == "square":
        nodes = [nd.equidistant_nodes(1, n)[0] for n in order[:2]]
    elif shape == "cube":
        nodes = [nd.equidistant_nodes(1, n)[0] for n in order[:3]]
    elif shape == "squared_cube":
        square = nd.tensor_product_nodes([
            nd.equidistant_nodes(1, n)[0] for n in order[:2]
            ])
        nodes = [square, nd.equidistant_nodes(1, order[2])[0]]
    elif shape == "prism":
        triangle = nd.warp_and_blend_nodes(2, order[0])
        nodes = [triangle, nd.equidistant_nodes(1, order[1])[0]]
    else:
        raise ValueError(f"unknown shape name: '{shape}'")

    nodes = nd.tensor_product_nodes(nodes)

    if not visualize or nodes.shape[0] <= 2:
        return

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(*nodes)
    for i in range(nodes.shape[1]):
        ax.text(*nodes[:, i], f"{i}")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    plt.show(block=True)

# }}}


# {{{ test_tensor_product_nodes_vs_tuples

def test_tensor_product_nodes_vs_tuples():
    import modepy as mp
    shapes = [
            (shp.Hypercube(2), (3, 5)),
            (shp.Hypercube(3), (3, 5, 4)),
            ]

    for shape, order in shapes:
        space = mp.space_for_shape(shape, order)
        ref_nodes = nd.equispaced_nodes_for_space(space, shape)
        nodes = (np.array(nd.node_tuples_for_space(space), dtype=np.float64)
                / np.array(order) * 2 - 1).T

        assert np.linalg.norm(nodes - ref_nodes) < 1.0e-14

# }}}


# {{{ test_random_nodes_for_tensor_product

def test_random_nodes_for_tensor_product():
    import modepy as mp
    shape = mp.TensorProductShape((mp.Simplex(1), mp.Simplex(2)))

    nnodes = 10
    nodes = mp.random_nodes_for_shape(shape, nnodes)
    assert nodes.shape == (3, 10)

    nnodes = 1
    nodes = mp.random_nodes_for_shape(shape, nnodes)
    assert nodes.shape == (3, 1)

# }}}


def test_tp_0d():
    import modepy as mp
    shape = mp.Hypercube(0)
    space = mp.QN(0, 5)

    for node_func in [
        mp.equispaced_nodes_for_space,
        mp.edge_clustered_nodes_for_space,
    ]:
        nodes = node_func(space, shape)
        assert nodes.shape == (0, 1)


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
