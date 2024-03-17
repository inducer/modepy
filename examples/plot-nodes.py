import matplotlib.pyplot as pt
import numpy as np

import modepy as mp
from modepy.tools import (
    EQUILATERAL_VERTICES, barycentric_to_equilateral, unit_to_barycentric)


dims = 3
n = 10

unit = mp.warp_and_blend_nodes(dims, n)

if False:
    from modepy.tools import estimate_lebesgue_constant
    lebesgue = estimate_lebesgue_constant(n, unit, visualize=True)

equi = barycentric_to_equilateral(unit_to_barycentric(unit))

if dims == 2:
    pt.plot(equi[0], equi[1], "o")

    uv = list(EQUILATERAL_VERTICES[2])
    uv.append(uv[0])
    uv = np.array(uv)
    pt.plot(uv[:, 0], uv[:, 1], "")

    pt.gca().set_aspect("equal")
    pt.show()
elif dims == 3:
    ax = pt.subplot(1, 1, 1, projection="3d")
    ax.plot(equi[0], equi[1], equi[2], "o")
    pt.show()
else:
    raise ValueError(f"Unsupported dimensions: {dims}")
