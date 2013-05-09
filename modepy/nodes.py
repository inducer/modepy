from __future__ import division

__copyright__ = "Copyright (C) 2009, 2010, 2013 Andreas Kloeckner, Tim Warburton, Jan Hesthaven, Xueyu Zhu"

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
from math import sqrt




__doc__ = """This module generates interpolation nodes as described in

    Warburton, T. "An Explicit Construction of Interpolation Nodes on the Simplex."
    Journal of Engineering Mathematics 56, no. 3 (2006): 247-262.
    http://dx.doi.org/10.1007/s10665-006-9086-6

The generated nodes have benign 
`Lebesgue constants <https://en.wikipedia.org/wiki/Lebesgue_constant_(interpolation)>`_.
"""




def get_1d_nodes(n, want_boundary_nodes):
    from modepy.quadrature.jacobi_gauss import (
            legendre_gauss_lobatto_nodes,
            LegendreGaussQuadrature)

    if want_boundary_nodes:
        return legendre_gauss_lobatto_nodes(n)
    else:
        return LegendreGaussQuadrature(n).nodes




def get_warp_factor(n, output_nodes, want_boundary_nodes=True, scaled=True):
    """Compute warp function at order *n* and evaluate it at
    the nodes *output_nodes*.
    """

    warped_nodes = get_1d_nodes(n, want_boundary_nodes=want_boundary_nodes)
    equi_nodes  = np.linspace(-1, 1, n+1)

    from modepy.matrices import vandermonde
    from modepy.modes import get_simplex_onb

    basis = get_simplex_onb(1, n)
    Veq = vandermonde(basis, equi_nodes)

    # create interpolator from equi_nodes to output_nodes
    eq_to_out = la.solve(Veq.T, vandermonde(basis, output_nodes).T).T

    # compute warp factor
    warp = np.dot(eq_to_out, warped_nodes - equi_nodes)
    if scaled:
        zerof = (abs(output_nodes)<1.0-1.0e-10)
        sf = 1.0 - (zerof*output_nodes)**2
        warp = warp/sf + warp*(zerof-1)

    return warp



# {{{ 2D nodes

_alpha_opt_2d = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,\
        1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223,1.6258]

def get_2d_warp_and_blend_nodes(n, want_boundary_nodes, node_tuples=None):
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
            equilateral_to_unit,
            EQUILATERAL_VERTICES)
    bary = unit_to_barycentric(unit_nodes)
    equi = barycentric_to_equilateral(bary)

    equi_vertices = EQUILATERAL_VERTICES[2]

    for i1 in range(3):
        i2, i3 = set(range(3)) - set([i1])

        # Compute blending function at each node for each edge
        blend = 4*bary[i2]*bary[i3]

        # Amount of warp for each node, for each edge
        warpf = get_warp_factor(n, bary[i2]-bary[i3])

        # Combine blend & warp
        warp = blend*warpf*(1 + (alpha*bary[i1])**2)

        # all vertices have the same distance from the origin
        tangent = equi_vertices[i2] - equi_vertices[i3]
        tangent /= la.norm(tangent)

        equi += tangent[:, np.newaxis] * warp[np.newaxis, :]

    return equilateral_to_unit(equi)

# }}}

# {{{ 3D nodes

def eval_warp(n, xnodes, xout):
    # Purpose: compute one-dimensional edge warping function

    warp = np.zeros((len(xout),1))
    xeq  = np.zeros((n+1,1))
    for i in range(n+1):
        xeq[i] = -1. + (2.*(n-i))/n;

    for i in range(n+1):
        d = xnodes[i]-xeq[i]

        for j in range(1,n):
            if i!=j:
                d = d*(xout-xeq[j])/(xeq[i]-xeq[j]);

        if i!=0:
            d = -d/(xeq[i]-xeq[0])

        if i!=n:
            d = d/(xeq[i]-xeq[n])

        warp = warp+d;

    return warp


def eval_shift(N, pval, L1, L2, L3):

    # Purpose: compute two-dimensional Warp & Blend transform

    # 1) compute Gauss-Lobatto-Legendre node distribution
    gaussX = -JacobiGL(0,0,N)

    # 3) compute blending function at each node for each edge
    blend1 = L2*L3
    blend2 = L1*L3
    blend3 = L1*L2

    # 4) amount of warp for each node, for each edge
    warpfactor1 = 4*evalwarp(N, gaussX, L3-L2)
    warpfactor2 = 4*evalwarp(N, gaussX, L1-L3)
    warpfactor3 = 4*evalwarp(N, gaussX, L2-L1)


    # 5) combine blend & warp
    warp1 = blend1*warpfactor1*(1 + (pval*L1)**2)
    warp2 = blend2*warpfactor2*(1 + (pval*L2)**2)
    warp3 = blend3*warpfactor3*(1 + (pval*L3)**2)

    # 6) evaluate shift in equilateral triangle
    dx = 1*warp1 + np.cos(2.*np.pi/3.)*warp2 + np.cos(4.*np.pi/3.)*warp3;
    dy = 0*warp1 + np.sin(2.*np.pi/3.)*warp2 + np.sin(4.*np.pi/3.)*warp3;

    return dx, dy


def  WarpShiftFace3D(p, pval, pval2, L1, L2, L3, L4):

    # Purpose: compute warp factor used in creating 3D Warp & Blend nodes

    dtan1,dtan2 = evalshift(p, pval, L2, L3, L4);

    warpx = dtan1
    warpy = dtan2

    return warpx, warpy

_alpha_opt_3d = [
        0, 0, 0, 0.1002,  1.1332, 1.5608, 1.3413, 1.2577, 1.1603,
        1.10153, 0.6080, 0.4523, 0.8856, 0.8717, 0.9655]

def get_3d_warp_and_blend_nodes(n, node_tuples=None):
    try:
        alpha = _alpha_opt_3d[n-1]
    except IndexError:
        alpha = 1.

    if node_tuples is None:
        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
                as gnitstam
        node_tuples = list(gnitstam(n, 2))
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

    for i1 in range(4):
        i2, i3, i4 = set(range(4)) - set([i1])

        l2,

        # all vertices have the same distance from the origin
        tangent1 = equi_vertices[i2] - equi_vertices[i3]
        tangent1 /= la.norm(tangent1)

        tangent2 = equi_vertices[i3] - equi_vertices[i4]
        tangent2 /= la.norm(tangent2)

        warp1, warp2 = eval_shift(n, alpha, La, Lb, Lc, Ld)




















































    r,s,t = EquiNodes3D(N)

    L1 = (1.+t)/2
    L2 = (1.+s)/2
    L3 = -(1.+r+s+t)/2
    L4 =  (1+r)/2

    # set vertices of tetrahedron
    v1 = np.array([-1., -1./sqrt(3.), -1./sqrt(6.)]) # row array
    v2 = np.array([ 1., -1./sqrt(3.), -1./sqrt(6.)])
    v3 = np.array([ 0,   2./sqrt(3.), -1./sqrt(6.)])
    v4 = np.array([ 0,            0,  3./sqrt(6.)])

    # orthogonal axis tangents on faces 1-4
    t1 = np.zeros((4,3))
    t1[0,:] = v2-v1
    t1[1,:] = v2-v1
    t1[2,:] = v3-v2
    t1[3,:] = v3-v1

    t2 = np.zeros((4,3))
    t2[0,:] = v3-0.5*(v1+v2)
    t2[1,:] = v4-0.5*(v1+v2)
    t2[2,:] = v4-0.5*(v2+v3)
    t2[3,:] = v4-0.5*(v1+v3)

    for n in range(4):
        # normalize tangents
        norm_t1 = la.norm(t1[n,:])
        norm_t2 = la.norm(t2[n,:])
        t1[n,:] = t1[n,:]/norm_t1 # 2-norm np.array ?
        t2[n,:] = t2[n,:]/norm_t2

    # Warp and blend for each face (accumulated in shiftXYZ)
    XYZ = L3*v1+L4*v2+L2*v3+L1*v4  # form undeformed coordinates
    shift = np.zeros((Np,3))
    for face in range(4):
        if(face==0):
            La = L1; Lb = L2; Lc = L3; Ld = L4;  # check  syntax

        if(face==1):
            La = L2; Lb = L1; Lc = L3; Ld = L4;

        if(face==2):
            La = L3; Lb = L1; Lc = L4; Ld = L2;

        if(face==3):
            La = L4; Lb = L1; Lc = L3; Ld = L2;

        #  compute warp tangential to face
        warp1, warp2 = eval_shift(N, alpha, alpha, La, Lb, Lc, Ld)

        # compute volume blending
        blend = Lb*Lc*Ld

        # modify linear blend
        denom = (Lb+0.5*La)*(Lc+0.5*La)*(Ld+0.5*La)
        ids = np.argwhere(denom>tol) # syntax
        ids = ids[:,0]

        blend[ids] = (1+(alpha*La[ids])**2)*blend[ids]/denom[ids]

        # compute warp & blend
        shift = shift + (blend*warp1)*t1[face,:]
        shift = shift + (blend*warp2)*t2[face,:]

        # fix face warp
        ids = np.argwhere((La<tol) *( (Lb>tol) + (Lc>tol) + (Ld>tol) < 3)) # syntax ??
        ids = ids[:,0]

        shift[ids,:] = warp1[ids]*t1[face,:] + warp2[ids]*t2[face,:]



    # shift nodes and extract individual coordinates
    XYZ = XYZ + shift
    x = XYZ[:,0]
    y = XYZ[:,1]
    z = XYZ[:,2]

    return x, y, z

# }}}

def get_warp_and_blend_nodes(dims, n, want_boundary_nodes, node_tuples=None):
    """
    :arg dims: dimensionality of desired simplex
        (1, 2 or 3, i.e. interval, triangle or tetrahedron).
    :arg n: Desired maximum total polynomial degree to interpolate.
    :arg node_tuples: a list of tuples of integers indicating the node order.
        Use default order if *None*, see
        :func:`pytools.generate_nonnegative_integer_tuples_summing_to_at_most`.
    :returns: An array of shape *(dims, nnodes)* containing unit coordinates
        of the interpolation nodes. (see :ref:`tri-coords` and :ref:`tet-coords`)
    """
    if dims == 1:
        if node_tuples is not None:
            raise NotImplementedError("specifying node_tuples in 1D")
        return get_1d_nodes(n, want_boundary_nodes=want_boundary_nodes)
    elif dims == 2:
        return get_2d_warp_and_blend_nodes(n, want_boundary_nodes, node_tuples)
    elif dims == 3:
        return get_3d_warp_and_blend_nodes(n, want_boundary_nodes, node_tuples)
    else:
        raise NotImplementedError("%d-dimensional bases" % dims)


# vim: foldmethod=marker
