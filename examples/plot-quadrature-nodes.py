import numpy as np
import matplotlib.pyplot as plt

import modepy as mp


def get_reference_element_nodes(name, dims):
    if dims == 2:
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        if name == "simplex":
            poly = Polygon([
                [-1, -1], [1, -1], [-1, 1]
                ], fill=None)
        elif name == "hypercube":
            poly = Polygon([
                [-1, -1], [1, -1], [1, 1], [-1, 1]
                ], fill=None)
        else:
            raise ValueError(f"unsupported reference element: '{name}'")

        patches = PatchCollection([poly],
                edgecolor="k", linewidth=4, match_original=True)
    elif dims == 3:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        if name == "simplex":
            patches = Line3DCollection([
                [[-1, -1, -1], [1, -1, -1]],
                [[1, -1, -1], [-1, 1, -1]],
                [[-1, 1, -1], [-1, -1, -1]],
                [[-1, -1, -1], [-1, -1, 1]],
                [[1, -1, -1], [-1, -1, 1]],
                [[-1, 1, -1], [-1, -1, 1]]
                ], colors="k")
        elif name == "hypercube":
            patches = Line3DCollection([
                [[-1, -1, -1], [1, -1, -1]],
                [[1, -1, -1], [1, 1, -1]],
                [[1, 1, -1], [-1, 1, -1]],
                [[-1, 1, -1], [-1, -1, -1]],

                [[-1, -1, 1], [1, -1, 1]],
                [[1, -1, 1], [1, 1, 1]],
                [[1, 1, 1], [-1, 1, 1]],
                [[-1, 1, 1], [-1, -1, 1]],

                [[-1, -1, -1], [-1, -1, 1]],
                [[1, -1, -1], [1, -1, 1]],
                [[1, 1, -1], [1, 1, 1]],
                [[-1, 1, -1], [-1, 1, 1]]
                ], colors="k")
        else:
            raise ValueError(f"unsupported reference element: '{name}'")
    else:
        return None

    return patches


def plot_quadrature_rule_nodes(name, order, dims, show=False):
    # {{{ nodes

    if name == "cc":
        quad_nodes = mp.ClenshawCurtisQuadrature(order).nodes
        ref_nodes = get_reference_element_nodes("line", 1)
    elif name == "gm":
        quad_nodes = mp.GrundmannMoellerSimplexQuadrature(order, dims).nodes
        ref_nodes = get_reference_element_nodes("simplex", dims)
    elif name == "vr":
        quad_nodes = mp.VioreanuRokhlinSimplexQuadrature(order, dims).nodes
        ref_nodes = get_reference_element_nodes("simplex", dims)
    elif name == "wv":
        quad_nodes = mp.WitherdenVincentQuadrature(order, dims).nodes
        ref_nodes = get_reference_element_nodes("hypercube", dims)
    elif name == "xg":
        quad_nodes = np.XiaoGimbutasSimplexQuadrature(order, dims).nodes
        ref_nodes = get_reference_element_nodes("simplex", dims)
    else:
        raise ValueError(f"unknown quadrature: '{name}'")

    if len(quad_nodes.shape) == 1:
        # handle one-dimensional case
        quad_nodes = np.vstack([
            np.linspace(-1.0, 1.0, quad_nodes.size),
            quad_nodes])

    # }}}

    # {{{ plot

    if dims == 3:
        from mpl_toolkits.mplot3d import Axes3D     # noqa: F401
        projection = "3d"
    else:
        projection = None

    fig = plt.figure()
    ax = fig.gca(projection=projection)
    ax.grid()

    if ref_nodes is not None:
        ax.add_collection(ref_nodes)
    ax.plot(*quad_nodes, "o", markersize=12)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    if dims == 3:
        ax.set_zlim([-1.1, 1.1])
        ax.view_init(15, 45)

    ax.set_title(f"${quad_nodes.shape[1]}~ nodes$")
    fig.savefig(f"quadrature_rule_{name}_order_{order}_{dims}d")
    if show:
        plt.show()

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="vr")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--dims", type=int, default=2)
    args = parser.parse_args()

    plot_quadrature_rule_nodes(args.name, args.order, args.dims)
