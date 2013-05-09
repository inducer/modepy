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




NODETOL = 1e-12
eps = np.finfo(float).eps



def Warpfactor(N, rout):
    """Compute scaled warp function at order N based on
    rout interpolation nodes.
    """

    # Compute LGL and equidistant node distribution
    LGLr = JacobiGL(0,0, N)
    req  = np.linspace(-1,1, N+1)
    # Compute V based on req
    Veq = Vandermonde1D(N, req)
    # Evaluate Lagrange polynomial at rout
    Nr = len(rout); Pmat = np.zeros((N+1, Nr))
    for i in range(N+1):
        Pmat[i,:] = JacobiP(rout.T[0,:], 0, 0, i)

    Lmat = la.solve(Veq.T, Pmat)

    # Compute warp factor
    warp = np.dot(Lmat.T, LGLr - req)
    warp = warp.reshape(Lmat.shape[1],1)
    zerof = (abs(rout)<1.0-1.0e-10)
    sf = 1.0 - (zerof*rout)**2
    warp = warp/sf + warp*(zerof-1)
    return warp

def Nodes2D(N):
    """Compute (x, y) nodes in equilateral triangle for polynomial
    of order N.
    """

    alpopt = np.array([0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,\
            1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223,1.6258])

    # Set optimized parameter, alpha, depending on order N
    if N< 16:
        alpha = alpopt[N-1]
    else:
        alpha = 5.0/3.0

    # total number of nodes
    Np = (N+1)*(N+2)//2

    # Create equidistributed nodes on equilateral triangle
    L1 = np.zeros((Np,1))
    L2 = np.zeros((Np,1))
    L3 = np.zeros((Np,1))
    sk = 0
    for n in range(N+1):
        for m in range(N+1-n):
            L1[sk] = n/N
            L3[sk] = m/N
            sk = sk+1

    L2 = 1.0-L1-L3
    x = -L2+L3; y = (-L2-L3+2*L1)/sqrt(3.0)

    # Compute blending function at each node for each edge
    blend1 = 4*L2*L3; blend2 = 4*L1*L3; blend3 = 4*L1*L2

    # Amount of warp for each node, for each edge
    warpf1 = Warpfactor(N, L3-L2)
    warpf2 = Warpfactor(N, L1-L3)
    warpf3 = Warpfactor(N, L2-L1)

    # Combine blend & warp
    warp1 = blend1*warpf1*(1 + (alpha*L1)**2)
    warp2 = blend2*warpf2*(1 + (alpha*L2)**2)
    warp3 = blend3*warpf3*(1 + (alpha*L3)**2)

    # Accumulate deformations associated with each edge
    x = x + 1*warp1 + np.cos(2*np.pi/3)*warp2 + np.cos(4*np.pi/3)*warp3
    y = y + 0*warp1 + np.sin(2*np.pi/3)*warp2 + np.sin(4*np.pi/3)*warp3
    return x, y


def evalwarp(N, xnodes, xout):

    # Purpose: compute one-dimensional edge warping function

    warp = np.zeros((len(xout),1))
    xeq  = np.zeros((N+1,1))
    for i in range(N+1):
        xeq[i] = -1. + (2.*(N-i))/N;

    for i in range(N+1):
        d = xnodes[i]-xeq[i]

        for j in range(1,N):
            if(i!=j):
                d = d*(xout-xeq[j])/(xeq[i]-xeq[j]);

        if(i!=0):
            d = -d/(xeq[i]-xeq[0])

        if(i!=N):
            d = d/(xeq[i]-xeq[N])

        warp = warp+d;

    return warp


def evalshift(N, pval, L1, L2, L3):

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

def EquiNodes3D(N):

    # Purpose: compute the equidistributed nodes on the
    #         reference tetrahedron

    # total number of nodes
    Np = (N+1)*(N+2)*(N+3)//6

    # 2) create equidistributed nodes on equilateral triangle
    X = np.zeros((Np,1))
    Y = np.zeros((Np,1))
    Z = np.zeros((Np,1))

    sk = 0
    for n in range(N+1):
        for m in range(N+1-n):
            for q in range(N+1-n-m):

                X[sk] = -1 + (q*2.)/N
                Y[sk] = -1 + (m*2.)/N
                Z[sk] = -1 + (n*2.)/N;

                sk = sk+1;

    return X, Y, Z

def Nodes3D(N):
    """Compute (x, y, z) nodes in equilateral tet for polynomial of degree N.
    """

    alpopt = np.array([0, 0, 0, 0.1002,  1.1332, 1.5608, 1.3413, 1.2577, 1.1603,\
                           1.10153, 0.6080, 0.4523, 0.8856, 0.8717, 0.9655])

    if(N<=15):
        alpha = alpopt[N-1]
    else:
        alpha = 1.

    # total number of nodes and tolerance
    Np = (N+1)*(N+2)*(N+3)//6
    tol = 1e-8

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
        warp1, warp2 = WarpShiftFace3D(N, alpha, alpha, La, Lb, Lc, Ld)

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



def xytors(x, y):
    """From (x, y) in equilateral triangle to (r, s) coordinates in standard triangle."""

    L1 = (np.sqrt(3.0)*y+1.0)/3.0
    L2 = (-3.0*x - np.sqrt(3.0)*y + 2.0)/6.0
    L3 = ( 3.0*x - np.sqrt(3.0)*y + 2.0)/6.0

    r = -L2 + L3 - L1; s = -L2 - L3 + L1
    return r, s

def xyztorst(x, y, z):

    # TO BE CONVERTED

    v1 = np.array([-1,-1/sqrt(3), -1/sqrt(6)]) # sqrt ?
    v2 = np.array([ 1,-1/sqrt(3), -1/sqrt(6)])
    v3 = np.array([ 0, 2/sqrt(3), -1/sqrt(6)])
    v4 = np.array([ 0, 0/sqrt(3),  3/sqrt(6)])

    # back out right tet nodes
    rhs = np.zeros((3, len(x)))
    rhs[0,:] = x
    rhs[1,:] = y
    rhs[2,:] = z

    tmp = np.zeros((3, 1))
    tmp[:,0] =  0.5*(v2+v3+v4-v1)
    rhs = rhs - tmp*np.ones((1,len(x)))

    A = np.zeros((3,3))
    A[:,0] = 0.5*(v2-v1)
    A[:,1] = 0.5*(v3-v1)
    A[:,2] = 0.5*(v4-v1)

    RST = la.solve(A,rhs)

    r = RST[0,:] # need to transpose ?
    s = RST[1,:] # need to transpose ?
    t = RST[2,:] # need to transpose ?

    return r, s, t


def rstoab(r, s):
    """Transfer from (r, s) -> (a, b) coordinates in triangle.
    """

    Np = len(r)
    a = np.zeros((Np,1))
    for n in range(Np):
        if s[n] != 1:
            a[n] = 2*(1+r[n])/(1-s[n])-1
        else:
            a[n] = -1

    b = s
    return a, b


def rsttoabc(r,s,t):

    """Transfer from (r,s,t) -> (a,b,c) coordinates in triangle
    """

    Np = len(r)
    tol = 1e-10

    a = np.zeros((Np,1))
    b = np.zeros((Np,1))
    c = np.zeros((Np,1))
    for n in range(Np):
        if abs(s[n]+t[n])>tol:
            a[n] = 2*(1+r[n])/(-s[n]-t[n])-1
        else:
            a[n] = -1

        if abs(t[n]-1.)>tol:
            b[n] = 2*(1+s[n])/(1-t[n])-1
        else:
            b[n] = -1

        c[n] = t[n]

    return a, b, c


