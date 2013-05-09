import matplotlib.pyplot as pt
import numpy as np

x = np.linspace(-1, 1, 200)
from modepy.nodes import get_warp_factor
for n in [1, 2, 4] + range(6, 30, 5):
    pt.plot(x,
            get_warp_factor(n, x, scaled=False),
            label="N=%d" % n)

pt.legend()
pt.show()
