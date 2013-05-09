
def Vandermonde1D(N, xp):
    """Initialize the 1D Vandermonde Matrix.
    V_{ij} = phi_j(xp_i)
    """

    Nx = np.int32(xp.shape[0])
    N  = np.int32(N)
    V1D = np.zeros((Nx, N+1))

    for j in range(N+1):
            V1D[:, j] = JacobiP(xp, 0, 0, j).T # give the tranpose of Jacobi.p

    return V1D




def Vandermonde2D(N, r, s):
    """Initialize the 2D Vandermonde Matrix,  V_{ij} = phi_j(r_i, s_i)
    """

    V2D = np.zeros((len(r),(N+1)*(N+2)/2))

    # Transfer to (a, b) coordinates
    a, b = rstoab(r, s)

    # build the Vandermonde matrix
    sk = 0

    for i in range(N+1):
        for j in range(N-i+1):
            V2D[:, sk] = Simplex2DP(a, b, i, j)
            sk = sk+1
    return V2D

def Vandermonde3D(N, r, s, t):
    """Initialize the 3D Vandermonde Matrix,  V_{ij} = phi_j(r_i, s_i, t_i)
    """

    print 'Np computed as ', ((N+1)*(N+2)*(N+3))//6

    V3D = np.zeros((len(r),((N+1)*(N+2)*(N+3))//6))

    # Transfer to (a, b) coordinates
    a, b, c = rsttoabc(r, s, t)

    # build the Vandermonde matrix
    sk = 0

    for i in range(N+1):
        for j in range(N+1-i):
            for k in range(N+1-i-j):
                V3D[:, sk] = Simplex3DP(a, b, c, i, j, k)
                sk = sk+1
    return V3D



def GradVandermonde2D(N, Np, r, s):
    """Initialize the gradient of the modal basis
    (i, j) at (r, s) at order N.
    """

    V2Dr = np.zeros((len(r), Np))
    V2Ds = np.zeros((len(r), Np))

    # find tensor-product coordinates
    a, b = rstoab(r, s)
    # Initialize matrices
    sk = 0
    for i in range(N+1):
        for j in range(N-i+1):
            V2Dr[:, sk], V2Ds[:, sk] = GradSimplex2DP(a, b, i, j)
            sk = sk+1
    return V2Dr, V2Ds


def GradVandermonde3D(N, Np, r, s, t):
    """Initialize the gradient of the modal basis
    (i, j, k) at (r, s, t) at order N.
    """

    V3Dr = np.zeros((len(r), Np))
    V3Ds = np.zeros((len(r), Np))
    V3Dt = np.zeros((len(r), Np))

    # find tensor-product coordinates
    a, b, c = rsttoabc(r, s, t)
    # Initialize matrices
    sk = 0
    for i in range(N+1):
        for j in range(N+1-i):
            for k in range(N+1-i-j):
                V3Dr[:, sk], V3Ds[:, sk], V3Dt[:, sk] = GradSimplex3DP(a, b, c, i, j, k)
                sk = sk+1

    return V3Dr, V3Ds, V3Dt


