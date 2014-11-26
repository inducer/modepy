from __future__ import absolute_import
import matplotlib.pyplot as pt
import numpy as np
import modepy as mp

dims = 3
n = 10

unit = mp.warp_and_blend_nodes(dims, n)

if 0:
    from modepy.tools import estimate_lebesgue_constant
    lebesgue = estimate_lebesgue_constant(n, unit, visualize=True)

from modepy.tools import unit_to_barycentric, barycentric_to_equilateral
equi = barycentric_to_equilateral(unit_to_barycentric(unit))

if dims == 2:
    pt.plot(equi[0], equi[1], "o")

    from modepy.tools import EQUILATERAL_VERTICES
    uv = list(EQUILATERAL_VERTICES[2])
    uv.append(uv[0])
    uv = np.array(uv)
    pt.plot(uv[:, 0], uv[:, 1], "")

    pt.gca().set_aspect("equal")
    pt.show()
elif dims == 3:
    import mayavi.mlab as mlab
    mlab.points3d(
            equi[0],
            equi[1],
            equi[2])
    mlab.orientation_axes()
    mlab.show()
