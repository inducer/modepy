import matplotlib.pyplot as pt
import numpy as np

x = np.linspace(-1, 1, 200)
from modepy.nodes import get_warp_factor
for n in [1, 2, 4] + range(6, 30, 5):
    pt.plot(x,
            get_warp_factor(n, x, want_boundary_nodes=True, scaled=False),
            label="N=%d Lobatto" % n)
    pt.plot(x,
            get_warp_factor(n, x, want_boundary_nodes=False, scaled=False), "--",
            label="N=%d Gauss" % n)

pt.legend()
pt.show()
