from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

import modepy as mp


if TYPE_CHECKING:
    from matplotlib.collections import Collection


def get_reference_element_nodes(
        name: str,
        dims: int
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], Collection | None]:
    patches: Collection | None

    if dims == 1:
        ref_vertices = np.array([[-1.0], [1.0]])
        patches = None
    elif dims == 2:
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        if name == "simplex":
            ref_vertices = np.array([[-1, -1], [1, -1], [-1, 1]])
        elif name == "hypercube":
            ref_vertices = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        else:
            raise ValueError(f"unsupported reference element: '{name}'")

        patches = PatchCollection(
            [Polygon(ref_vertices, fill=None)],
            edgecolor="#4C72B0", facecolor="#D2DBEB", linewidth=4, match_original=True)
    elif dims == 3:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        if name == "simplex":
            ref_vertices = np.array([
                [[-1, -1, -1], [1, -1, -1]],
                [[1, -1, -1], [-1, 1, -1]],
                [[-1, 1, -1], [-1, -1, -1]],
                [[-1, -1, -1], [-1, -1, 1]],
                [[1, -1, -1], [-1, -1, 1]],
                [[-1, 1, -1], [-1, -1, 1]]
                ])
        elif name == "hypercube":
            ref_vertices = np.array([
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
                ])
        else:
            raise ValueError(f"unsupported reference element: '{name}'")

        patches = Line3DCollection(ref_vertices, colors="#4C72B0")
    else:
        raise ValueError(f"Unsupported dimension: {dims}")

    return ref_vertices, patches


def plot_quadrature_rule_nodes(
        name: str,
        order: int,
        dims: int, *,
        hide_axis: bool = False,
        show: bool = False) -> None:
    # {{{ nodes

    if name == "cc":
        quad_nodes = mp.ClenshawCurtisQuadrature(order, force_dim_axis=True).nodes
        ref_vertices, patches = get_reference_element_nodes("line", 1)
    elif name == "gm":
        quad_nodes = mp.GrundmannMoellerSimplexQuadrature(order, dims).nodes
        ref_vertices, patches = get_reference_element_nodes("simplex", dims)
    elif name == "js":
        quad_nodes = mp.JaskowiecSukumarQuadrature(order, dims).nodes
        ref_vertices, patches = get_reference_element_nodes("simplex", dims)
    elif name == "vr":
        quad_nodes = mp.VioreanuRokhlinSimplexQuadrature(order, dims).nodes
        ref_vertices, patches = get_reference_element_nodes("simplex", dims)
    elif name == "wv":
        quad_nodes = mp.WitherdenVincentQuadrature(order, dims).nodes
        ref_vertices, patches = get_reference_element_nodes("hypercube", dims)
    elif name == "xg":
        quad_nodes = mp.XiaoGimbutasSimplexQuadrature(order, dims).nodes
        ref_vertices, patches = get_reference_element_nodes("simplex", dims)
    elif name == "sc":  # simplex edge-clustered nodes
        quad_nodes = mp.warp_and_blend_nodes(dims, order)
        ref_vertices, patches = get_reference_element_nodes("simplex", dims)
    elif name == "qc":  # quad edge-clustered nodes
        quad_nodes = mp.legendre_gauss_lobatto_tensor_product_nodes(dims, order)
        ref_vertices, patches = get_reference_element_nodes("hypercube", dims)
    elif name == "se":  # simplex equidistant nodes
        quad_nodes = mp.equidistant_nodes(dims, order)
        ref_vertices, patches = get_reference_element_nodes("simplex", dims)
    elif name == "qe":  # quad equidistant nodes
        quad_nodes = mp.tensor_product_nodes([
            mp.equidistant_nodes(1, order) for _ in range(dims)
            ])
        ref_vertices, patches = get_reference_element_nodes("hypercube", dims)
    else:
        raise ValueError(f"unknown quadrature: '{name}'")

    if len(quad_nodes.shape) == 1:
        # handle one-dimensional case
        quad_nodes = np.vstack([
            np.linspace(-1.0, 1.0, quad_nodes.size),
            quad_nodes])

    # }}}

    # {{{ plot

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection="3d" if dims == 3 else None)

    ax.plot(*ref_vertices.T, "o", color="#4C72B0", markersize=14)
    ax.plot(*quad_nodes, "o", color="k", markersize=10)
    if patches is not None:
        ax.add_collection(patches)

    ax.grid()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    if dims == 3:
        ax.set_zlim([-1.1, 1.1])
        ax.view_init(15, 45)

    if hide_axis:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
    else:
        ax.set_title(f"${quad_nodes.shape[1]}~ nodes$")

    fig.savefig(f"quadrature_rule_{name}_order_{order}_{dims}d",
                pad_inches=0,
                bbox_inches="tight")
    if show:
        plt.show()

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name",
                        choices=["cc", "gm", "js", "vr", "wv", "xg",
                                 "sc", "qc", "se", "qe"],
                        default="vr")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--hide-axis", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    plot_quadrature_rule_nodes(args.name, args.order, args.dims,
                               hide_axis=args.hide_axis,
                               show=args.show)
