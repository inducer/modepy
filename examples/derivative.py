import numpy as np
import modepy as mp

n = 17 # use this total degree
dimensions = 2

# Get a basis of orthonormal functions, and their derivatives.

basis = mp.simplex_onb(dimensions, n)
grad_basis = mp.grad_simplex_onb(dimensions, n)

nodes = mp.warp_and_blend_nodes(dimensions, n)
x, y = nodes

# We want to compute the x derivative of this function:

f = np.sin(5*x+y)
df_dx = 5*np.cos(5*x+y)

# The (generalized) Vandermonde matrix transforms coefficients into
# nodal values. So we can find basis coefficients by applying its
# inverse:

f_coefficients = np.linalg.solve(
        mp.vandermonde(basis, nodes), f)

# Now linearly combine the (x-)derivatives of the basis using
# f_coefficients to compute the numerical derivatives.

df_dx_num = np.dot(
        mp.vandermonde(grad_basis, nodes)[0], f_coefficients)

assert np.linalg.norm(df_dx - df_dx_num) < 1e-5
