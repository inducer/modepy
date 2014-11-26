from __future__ import absolute_import
import numpy as np
from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
        as gnitstam
from six.moves import zip

# prepare plot and eval nodes on triangle
dims = 2
node_n = 40
node_tuples = list(gnitstam(node_n, dims))
plot_nodes = np.array(node_tuples, dtype=np.float64) / node_n
eval_nodes = 2*(plot_nodes - 0.5).T

# get triangle submesh
from modepy.tools import submesh
tri_subtriangles = np.array(submesh(node_tuples))

# evaluate each basis function, build global tri mesh
node_count = 0
all_nodes = []
all_triangles = []
all_values = []

from modepy.modes import simplex_onb

p = 3
stretch_factor = 1.5

for (i, j), basis_func in zip(
        gnitstam(p, dims),
        simplex_onb(dims, p),
        ):

    all_nodes.append(plot_nodes + [stretch_factor*i, stretch_factor*j])
    all_triangles.append(tri_subtriangles + node_count)
    all_values.append(basis_func(eval_nodes))
    node_count += len(plot_nodes)

all_nodes = np.vstack(all_nodes)
all_triangles = np.vstack(all_triangles)
all_values = np.hstack(all_values)

# plot
import mayavi.mlab as mlab
fig = mlab.figure(bgcolor=(1, 1, 1))
mlab.triangular_mesh(
        all_nodes[:, 0],
        all_nodes[:, 1],
        0.2*all_values,
        all_triangles)

x, y = np.mgrid[-1:p*stretch_factor +1:20j, -1:p*stretch_factor +1:20j]
mlab.mesh(x, y, 0*x, representation="wireframe", color=(0.4, 0.4, 0.4), line_width=0.6)

mlab.view(-153, 58, 10, np.array([ 1.61,  2.49, -0.59]))

mlab.show()
