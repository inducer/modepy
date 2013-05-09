import matplotlib.pyplot as pt
import numpy as np

from modepy.nodes import get_warp_and_blend_nodes
r, s = get_warp_and_blend_nodes(2, 10, want_boundary_nodes=False)

pt.plot(r, s, "o")

from modepy.tools import UNIT_VERTICES
uv = list(UNIT_VERTICES[2])
uv.append(uv[0])
uv = np.array(uv)
pt.plot(uv[:, 0], uv[:, 1], "")
pt.show()
