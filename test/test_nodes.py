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




import numpy as np
import numpy.linalg as la
import pytest




def test_warp():
    """Check some assumptions on the node warp factor calculator"""
    n = 17
    from modepy.discretization.local import WarpFactorCalculator
    wfc = WarpFactorCalculator(n)

    assert abs(wfc.int_f(-1)) < 1e-12
    assert abs(wfc.int_f(1)) < 1e-12

    from modepy.quadrature import LegendreGaussQuadrature

    lgq = LegendreGaussQuadrature(n)
    assert abs(lgq(wfc)) < 6e-14




def test_tri_nodes_against_known_values():
    """Check triangle nodes against a previous implementation"""
    from modepy.discretization.local import TriangleDiscretization
    triorder = 8
    tri = TriangleDiscretization(triorder)

    def tri_equilateral_nodes_reference(self):
        # This is the old, more explicit, less general way of computing
        # the triangle nodes. Below, we compare its results with that of the
        # new routine.

        alpha_opt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
                1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]

        try:
            alpha = alpha_opt[self.order-1]
        except IndexError:
            alpha = 5/3

        from modepy.discretization.local import WarpFactorCalculator
        from math import sin, cos, pi

        warp = WarpFactorCalculator(self.order)

        edge1dir = np.array([1,0])
        edge2dir = np.array([cos(2*pi/3), sin(2*pi/3)])
        edge3dir = np.array([cos(4*pi/3), sin(4*pi/3)])

        for bary in self.equidistant_barycentric_nodes():
            lambda1, lambda2, lambda3 = bary

            # find equidistant (x,y) coordinates in equilateral triangle
            point = self.barycentric_to_equilateral(bary)

            # compute blend factors
            blend1 = 4*lambda1*lambda2 # nonzero on AB
            blend2 = 4*lambda3*lambda2 # nonzero on BC
            blend3 = 4*lambda3*lambda1 # nonzero on AC

            # calculate amount of warp for each node, for each edge
            warp1 = blend1*warp(lambda2 - lambda1)*(1 + (alpha*lambda3)**2)
            warp2 = blend2*warp(lambda3 - lambda2)*(1 + (alpha*lambda1)**2)
            warp3 = blend3*warp(lambda1 - lambda3)*(1 + (alpha*lambda2)**2)

            # return warped point
            yield point + warp1*edge1dir + warp2*edge2dir + warp3*edge3dir

    if False:
        outf = open("trinodes1.dat", "w")
        for ux in tri.equilateral_nodes():
            outf.write("%g\t%g\n" % tuple(ux))
        outf = open("trinodes2.dat", "w")
        for ux in tri_equilateral_nodes_reference(tri):
            outf.write("%g\t%g\n" % tuple(ux))

    for n1, n2 in zip(tri.equilateral_nodes(),
            tri_equilateral_nodes_reference(tri)):
        assert la.norm(n1-n2) < 3e-15

    def node_indices_2(order):
        for n in range(0, order+1):
             for m in range(0, order+1-n):
                 yield m,n

    assert set(tri.node_tuples()) == set(node_indices_2(triorder))





def test_tri_face_node_distribution():
    """Test whether the nodes on the faces of the triangle are distributed
    according to the same proportions on each face.

    If this is not the case, then reusing the same face mass matrix
    for each face would be invalid.
    """

    from modepy.discretization.local import TriangleDiscretization

    tri = TriangleDiscretization(8)
    unodes = tri.unit_nodes()
    projected_face_points = []
    for face_i in tri.face_indices():
        start = unodes[face_i[0]]
        end = unodes[face_i[-1]]
        dir = end-start
        dir /= np.dot(dir, dir)
        pfp = np.array([np.dot(dir, unodes[i]-start) for i in face_i])
        projected_face_points.append(pfp)

    first_points =  projected_face_points[0]
    for points in projected_face_points[1:]:
        error = la.norm(points-first_points, np.Inf)
        assert error < 1e-15




def test_simp_nodes():
    """Verify basic assumptions on simplex interpolation nodes"""
    from modepy.discretization.local import \
            IntervalDiscretization, \
            TriangleDiscretization, \
            TetrahedronDiscretization

    els = [
            IntervalDiscretization(19),
            TriangleDiscretization(8),
            TriangleDiscretization(17),
            TetrahedronDiscretization(13)]

    for el in els:
        eps = 1e-10

        unodes = list(el.unit_nodes())
        assert len(unodes) == el.node_count()
        for ux in unodes:
            for uc in ux:
                assert uc >= -1-eps
            assert sum(ux) <= 1+eps

        try:
            equnodes = list(el.equidistant_unit_nodes())
        except AttributeError:
            assert isinstance(el, IntervalDiscretization)
        else:
            assert len(equnodes) == el.node_count()
            for ux in equnodes:
                for uc in ux:
                    assert uc >= -1-eps
                assert sum(ux) <= 1+eps

        for indices in el.node_tuples():
            for index in indices:
                assert index >= 0
            assert sum(indices) <= el.order





def test_affine_map():
    """Check that our cheapo geometry-targeted linear algebra actually works."""
    from modepy.tools import AffineMap
    for d in range(1, 5):
    #for d in [3]:
        for i in range(100):
            a = np.random.randn(d, d)+10*np.eye(d)
            b = np.random.randn(d)

            m = AffineMap(a, b)

            x = np.random.randn(d, 10)


            assert la.norm(x-m.inverse(m(x))) < 1e-10




def test_tri_map():
    """Verify that the mapping and node-building operations maintain triangle vertices"""
    from modepy.discretization.local import TriangleDiscretization

    n = 8
    tri = TriangleDiscretization(n)

    node_dict = dict((ituple, idx) for idx, ituple in enumerate(tri.node_tuples()))
    corner_indices = [node_dict[0,0], node_dict[n,0], node_dict[0,n]]
    unodes = tri.unit_nodes()
    corners = [unodes[i] for i in corner_indices]

    for i in range(10):
        vertices = [np.random.randn(2) for vi in range(3)]
        map = tri.geometry.get_map_unit_to_global(vertices)
        global_corners = [map(pt) for pt in corners]
        for gc, v in zip(global_corners, vertices):
            assert la.norm(gc-v) < 1e-12




# You can test individual routines by typing
# $ python test_nodes.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
