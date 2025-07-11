from __future__ import annotations

import numpy as np

import modepy as mp


# Define the shape on which we will operate
line = mp.Simplex(1)
triangle = mp.Simplex(2)
prism = mp.TensorProductShape((triangle, line))

assert prism.dim == 3

# Define a function space for the prism
n = 12
space = mp.TensorProductSpace((mp.PN(triangle.dim, n), mp.PN(line.dim, n)))

assert space.order == n
assert space.spatial_dim == 3
assert space.space_dim == (n + 1) * (n + 1) * (n + 2) // 2

# Define a basis function for the prism
basis = mp.orthonormal_basis_for_space(space, prism)

# Define a point set for the prism
nodes = mp.edge_clustered_nodes_for_space(space, prism)

# Define a quadrature rule for the prism
quadrature = mp.TensorProductQuadrature([
    mp.VioreanuRokhlinSimplexQuadrature(n, triangle.dim),
    mp.LegendreGaussQuadrature(n),
])

# Define a bilinear form: weak derivative in the x direction
i = 0
weak_d = mp.nodal_quadrature_bilinear_form_matrix(
    quadrature,
    test_functions=basis.derivatives(i),
    trial_functions=basis.functions,
    nodal_interp_functions_test=basis.functions,
    nodal_interp_functions_trial=basis.functions,
    input_nodes=nodes,
    output_nodes=nodes,
)
inv_mass = mp.inverse_mass_matrix(basis, nodes)

# Compute derivative
f = 1.0 - np.sin(nodes[i]) ** 3
f_ref = -3.0 * np.cos(nodes[i]) * np.sin(nodes[i]) ** 2
f_approx = inv_mass @ weak_d.T @ f

error = np.linalg.norm(f_approx - f_ref) / np.linalg.norm(f_ref)
assert error < 1.0e-4
