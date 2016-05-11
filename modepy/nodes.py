from __future__ import division

__copyright__ = "Copyright (C) 2009, 2010, 2013 Andreas Kloeckner, " \
        "Tim Warburton, Jan Hesthaven, Xueyu Zhu"

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


def equidistant_nodes(dims, n, node_tuples=None):
    """
    :arg dims: dimensionality of desired simplex
        (e.g. 1, 2 or 3, for interval, triangle or tetrahedron).
    :arg n: Desired maximum total polynomial degree to interpolate.
    :arg node_tuples: a list of tuples of integers indicating the node order.
        Use default order if *None*, see
        :func:`pytools.generate_nonnegative_integer_tuples_summing_to_at_most`.
    :returns: An array of shape *(dims, nnodes)* containing unit coordinates
        of the interpolation nodes. (see :ref:`tri-coords` and :ref:`tet-coords`)
    """

    if node_tuples is None:
        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
                as gnitstam
        node_tuples = list(gnitstam(n, dims))
    else:
        if len(node_tuples) != (n+1)*(n+2)//2:
            raise ValueError("node_tuples list does not have the correct length")

    # shape: (2, nnodes)
    return (np.array(node_tuples, dtype=np.float64)/n*2 - 1).T


def warp_factor(n, output_nodes, scaled=True):
    """Compute warp function at order *n* and evaluate it at
    the nodes *output_nodes*.
    """

    from modepy.quadrature.jacobi_gauss import legendre_gauss_lobatto_nodes

    warped_nodes = legendre_gauss_lobatto_nodes(n)
    equi_nodes = np.linspace(-1, 1, n+1)

    from modepy.matrices import vandermonde
    from modepy.modes import simplex_onb

    basis = simplex_onb(1, n)
    Veq = vandermonde(basis, equi_nodes)

    # create interpolator from equi_nodes to output_nodes
    eq_to_out = la.solve(Veq.T, vandermonde(basis, output_nodes).T).T

    # compute warp factor
    warp = np.dot(eq_to_out, warped_nodes - equi_nodes)
    if scaled:
        zerof = (abs(output_nodes) < 1.0-1.0e-10)
        sf = 1.0 - (zerof*output_nodes)**2
        warp = warp/sf + warp*(zerof-1)

    return warp


# {{{ 2D nodes

def _2d_equilateral_shift(n, bary, alpha):
    from modepy.tools import EQUILATERAL_VERTICES
    equi_vertices = EQUILATERAL_VERTICES[2]

    result = np.zeros((2, bary.shape[1]))

    for i1 in range(3):
        i2, i3 = set(range(3)) - set([i1])

        # Compute blending function at each node for each edge
        blend = 4*bary[i2]*bary[i3]

        # Amount of warp for each node, for each edge
        warpf = warp_factor(n, bary[i2]-bary[i3])

        # Combine blend & warp
        warp = blend*warpf*(1 + (alpha*bary[i1])**2)

        # all vertices have the same distance from the origin
        tangent = equi_vertices[i2] - equi_vertices[i3]
        tangent /= la.norm(tangent)

        result += tangent[:, np.newaxis] * warp[np.newaxis, :]

    return result


_alpha_opt_2d = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
        1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]


def warp_and_blend_nodes_2d(n, node_tuples=None):
    try:
        alpha = _alpha_opt_2d[n-1]
    except IndexError:
        alpha = 5/3

    if node_tuples is None:
        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
                as gnitstam
        node_tuples = list(gnitstam(n, 2))
    else:
        if len(node_tuples) != (n+1)*(n+2)//2:
            raise ValueError("node_tuples list does not have the correct length")

    # shape: (2, nnodes)
    unit_nodes = (np.array(node_tuples, dtype=np.float64)/n*2 - 1).T

    from modepy.tools import (
            unit_to_barycentric,
            barycentric_to_equilateral,
            equilateral_to_unit)
    bary = unit_to_barycentric(unit_nodes)

    return equilateral_to_unit(
        barycentric_to_equilateral(bary)
        + _2d_equilateral_shift(n, bary, alpha))

# }}}


# {{{ 3D nodes

_alpha_opt_3d = [
        0, 0, 0, 0.1002,  1.1332, 1.5608, 1.3413, 1.2577, 1.1603,
        1.10153, 0.6080, 0.4523, 0.8856, 0.8717, 0.9655]


def warp_and_blend_nodes_3d(n, node_tuples=None):
    try:
        alpha = _alpha_opt_3d[n-1]
    except IndexError:
        alpha = 1.

    if node_tuples is None:
        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
                as gnitstam
        node_tuples = list(gnitstam(n, 3))
    else:
        if len(node_tuples) != (n+1)*(n+2)*(n+3)//6:
            raise ValueError("node_tuples list does not have the correct length")

    # shape: (3, nnodes)
    unit_nodes = (np.array(node_tuples, dtype=np.float64)/n*2 - 1).T

    from modepy.tools import (
            unit_to_barycentric,
            barycentric_to_equilateral,
            equilateral_to_unit,
            EQUILATERAL_VERTICES)
    bary = unit_to_barycentric(unit_nodes)
    equi = barycentric_to_equilateral(bary)

    equi_vertices = EQUILATERAL_VERTICES[3]

    # total number of nodes and tolerance
    tol = 1e-8

    shift = np.zeros_like(equi)

    for i1, i2, i3, i4, vertex_step in [
            (0, 1, 2, 3, -1),
            (1, 2, 3, 0, -1),
            (2, 3, 0, 1, -1),
            (3, 0, 1, 2, -1),
            ]:

        vi2, vi3, vi4 = [(i1 + vertex_step*i) % 4 for i in range(1, 4)]

        # all vertices have the same distance from the origin
        tangent1 = equi_vertices[vi3] - equi_vertices[vi4]
        tangent1 /= la.norm(tangent1)

        tangent2 = equi_vertices[vi2] - equi_vertices[vi3]
        tangent2 -= np.dot(tangent1, tangent2)*tangent1

        tangent2 /= la.norm(tangent2)

        sub_bary = bary[[i2, i3, i4]]
        warp1, warp2 = _2d_equilateral_shift(n, sub_bary, alpha)

        l1 = bary[i1]
        l2, l3, l4 = sub_bary

        blend = l2*l3*l4

        denom = (l2+0.5*l1)*(l3+0.5*l1)*(l4+0.5*l1)
        denom_ok = denom > tol

        blend[denom_ok] = (
                (1+(alpha*l1[denom_ok])**2)
                * blend[denom_ok]
                / denom[denom_ok])

        shift = shift + (blend*warp1)[np.newaxis, :] * tangent1[:, np.newaxis]
        shift = shift + (blend*warp2)[np.newaxis, :] * tangent2[:, np.newaxis]

        is_face = (l1 < tol) & ((l2 > tol) | (l3 > tol) | (l4 > tol))

        # assign face warp separately
        shift[:, is_face] = (
                + warp1[is_face][np.newaxis, :] * tangent1[:, np.newaxis]
                + warp2[is_face][np.newaxis, :] * tangent2[:, np.newaxis]
                )

    return equilateral_to_unit(equi + shift)

# }}}


def warp_and_blend_nodes(dims, n, node_tuples=None):
    """Return interpolation nodes as described in [warburton-nodes]_

    .. [warburton-nodes] Warburton, T.
        "An Explicit Construction of Interpolation Nodes on the Simplex."
        Journal of Engineering Mathematics 56, no. 3 (2006): 247-262.
        http://dx.doi.org/10.1007/s10665-006-9086-6

    :arg dims: dimensionality of desired simplex
        (1, 2 or 3, i.e. interval, triangle or tetrahedron).
    :arg n: Desired maximum total polynomial degree to interpolate.
    :arg node_tuples: a list of tuples of integers indicating the node order.
        Use default order if *None*, see
        :func:`pytools.generate_nonnegative_integer_tuples_summing_to_at_most`.
    :returns: An array of shape *(dims, nnodes)* containing unit coordinates
        of the interpolation nodes. (see :ref:`tri-coords` and :ref:`tet-coords`)

    The generated nodes have benign
    `Lebesgue constants
    <https://en.wikipedia.org/wiki/Lebesgue_constant_(interpolation)>`_.
    (See also :func:`modepy.tools.estimate_lebesgue_constant`)
    """
    if dims == 0:
        return np.empty((0, 1), dtype=np.float64)

    elif dims == 1:
        from modepy.quadrature.jacobi_gauss import legendre_gauss_lobatto_nodes
        result = legendre_gauss_lobatto_nodes(n)

        if node_tuples is not None:
            new_result = np.empty_like(result)
            if len(node_tuples) != n+1:
                raise ValueError("node_tuples list does not have the correct length")
            for i, (nti,) in enumerate(node_tuples):
                new_result[i] = result[nti]
            result = new_result

        return result.reshape(1, -1)

    elif dims == 2:
        return warp_and_blend_nodes_2d(n, node_tuples)

    elif dims == 3:
        return warp_and_blend_nodes_3d(n, node_tuples)

    else:
        raise NotImplementedError("%d-dimensional node sets" % dims)


# vim: foldmethod=marker
