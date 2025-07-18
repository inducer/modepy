from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import modepy as mp


# prepare basis vectors
shape = mp.Simplex(2)
space = mp.PN(2, 3)
basis = mp.orthonormal_basis_for_space(space, shape)

# prepare plot and eval nodes on triangle
plot_space = mp.PN(2, 32)
node_tuples = mp.node_tuples_for_space(plot_space)
eval_nodes = mp.equispaced_nodes_for_space(plot_space, shape)
plot_nodes = (eval_nodes.T + 1) / 2

# get triangle submesh
tri_subtriangles = np.array(mp.submesh_for_shape(shape, node_tuples))

# evaluate each basis function, build global triangle mesh
node_count = 0
all_nodes = []
all_triangles = []
all_values = []

stretch_factor = 1.5

for (i, j), basis_func in zip(basis.mode_ids, basis.functions, strict=True):
    all_nodes.append(plot_nodes + [stretch_factor * i, stretch_factor * j])  # noqa: RUF005
    all_triangles.append(tri_subtriangles + node_count)
    all_values.append(basis_func(eval_nodes))
    node_count += len(plot_nodes)

all_nodes = np.vstack(all_nodes)
all_triangles = np.vstack(all_triangles)
all_values = np.hstack(all_values)

# plot
x, y = np.mgrid[-1:space.order*stretch_factor + 1:20j,
                -1:space.order*stretch_factor + 1:20j]

fig = plt.figure(figsize=(10, 5), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.view_init(azim=-150, elev=25, roll=0)

ax.plot_wireframe(x, y, 0 * x, color="k", alpha=0.15, lw=0.5)
ax.plot_trisurf(all_nodes[:, 0], all_nodes[:, 1], 0.25 * all_values,
                triangles=all_triangles,
                cmap="jet",
                antialiased=False,
                edgecolor="none")

ax.set_axis_off()
ax.set_aspect("equal")
plt.savefig("plot-basis-pkdo-2d.png",
            transparent=True,
            pad_inches=0,
            bbox_inches="tight")
plt.show()
