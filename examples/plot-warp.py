import matplotlib.pyplot as pt
import numpy as np


x = np.linspace(-1, 1, 200)
from modepy.nodes import warp_factor


for n in [1, 2, 4] + list(range(6, 30, 5)):
    pt.plot(x,
            warp_factor(n, x, scaled=False),
            label="N=%d" % n)

pt.legend()
pt.show()
