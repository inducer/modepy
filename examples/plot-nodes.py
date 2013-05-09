import matplotlib.pyplot as pt
import numpy as np

from modepy.nodes import get_warp_and_blend_nodes
n = 15
unit = get_warp_and_blend_nodes(2, n, want_boundary_nodes=True)

from modepy.tools import estimate_lebesgue_constant
lebesgue = estimate_lebesgue_constant(n, unit, visualize=True)

from modepy.tools import unit_to_barycentric, barycentric_to_equilateral
x, y = barycentric_to_equilateral(unit_to_barycentric(unit))

pt.plot(x, y, "o")

from modepy.tools import EQUILATERAL_VERTICES
uv = list(EQUILATERAL_VERTICES[2])
uv.append(uv[0])
uv = np.array(uv)
pt.plot(uv[:, 0], uv[:, 1], "")

pt.gca().set_aspect("equal")
pt.show()
