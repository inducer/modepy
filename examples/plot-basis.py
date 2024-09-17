from __future__ import annotations

import matplotlib.pyplot as pt
import numpy as np

from pytools import (
    generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
)

from modepy.modes import simplex_onb
from modepy.tools import submesh


# prepare plot and eval nodes on triangle
dims = 2
node_n = 40
node_tuples = list(gnitstam(node_n, dims))
plot_nodes = np.array(node_tuples, dtype=np.float64) / node_n
eval_nodes = 2*(plot_nodes - 0.5).T

# get triangle submesh
tri_subtriangles = np.array(submesh(node_tuples))

# evaluate each basis function, build global tri mesh
node_count = 0
all_nodes = []
all_triangles = []
all_values = []

p = 3
stretch_factor = 1.5

for (i, j), basis_func in zip(
        gnitstam(p, dims),
        simplex_onb(dims, p),
        ):

    all_nodes.append([*plot_nodes, stretch_factor * i, stretch_factor * j])
    all_triangles.append(tri_subtriangles + node_count)
    all_values.append(basis_func(eval_nodes))
    node_count += len(plot_nodes)

all_nodes = np.vstack(all_nodes)
all_triangles = np.vstack(all_triangles)
all_values = np.hstack(all_values)

# plot
x, y = np.mgrid[-1:p*stretch_factor + 1:20j, -1:p*stretch_factor + 1:20j]

ax = pt.subplot(1, 1, 1, projection="3d")
ax.view_init(azim=-150, elev=25, roll=0)

ax.plot_wireframe(x, y, 0 * x, color="k", alpha=0.15, lw=0.5)
ax.plot_trisurf(all_nodes[:, 0], all_nodes[:, 1], 0.25 * all_values,
                triangles=all_triangles,
                cmap="jet",
                antialiased=False,
                edgecolor="none")

ax.set_axis_off()
ax.set_aspect("equal")
pt.savefig("plot-basis-pkdo-2d.png",
           transparent=True,
           dpi=300,
           pad_inches=0,
           bbox_inches="tight")
pt.show()
