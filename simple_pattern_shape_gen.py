import json
import numpy as np


def gen_shape(radius, shape):
    xi, yi, zi = np.meshgrid(
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        indexing="ij",
    )
    sphsrc = None
    if shape == "sphere":
        sphsrc = (xi**2 + yi**2 + zi**2) <= radius**2
        sphsrc = sphsrc.astype(np.int32)
        print("!!!!!", sphsrc)
        sphsrc = sphsrc.astype(np.float32)
        print("!!!!!qqq", sphsrc.max())

    return sphsrc, [radius, radius, radius]
