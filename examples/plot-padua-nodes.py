from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import modepy as mp


if TYPE_CHECKING:
    from modepy.typing import ArrayF


def _first_padua_curve(n: int, t: ArrayF) -> ArrayF:
    return np.vstack([-np.cos((n + 1) * t), -np.cos(n * t)])


def _second_padua_curve(n: int, t: ArrayF) -> ArrayF:
    return np.vstack([-np.cos(n * t), -np.cos((n + 1) * t)])


def _third_padua_curve(n: int, t: ArrayF) -> ArrayF:
    return np.vstack([np.cos((n + 1) * t), np.cos(n * t)])


def _fourth_padua_curve(n: int, t: ArrayF) -> ArrayF:
    return np.vstack([np.cos(n * t), np.cos((n + 1) * t)])


def plot_padua_nodes(alpha: float, beta: float, order: int, family: str) -> None:
    if family == "first":
        curve_fn = _first_padua_curve
    elif family == "second":
        curve_fn = _second_padua_curve
    elif family == "third":
        curve_fn = _third_padua_curve
    elif family == "fourth":
        curve_fn = _fourth_padua_curve
    else:
        raise ValueError(f"Unknown Padua curve family: '{family}'")

    t = np.linspace(0, np.pi, 512)
    curve = curve_fn(order, t)
    nodes = mp.padua_jacobi_nodes(alpha, beta, order, family)
    assert nodes.shape[1] == ((order + 1) * (order + 2) // 2)

    fig = plt.figure()
    ax = fig.gca()
    ax.grid()

    ax.plot(curve[0], curve[1])
    for i, xi in enumerate(nodes.T):
        ax.plot(nodes[0], nodes[1], "ko", markersize=8)
        ax.text(*xi, str(i), color="k", fontsize=24, fontweight="bold")

    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_aspect("equal")
    fig.savefig(f"padua_nodes_order_{order:02d}_family_{family}")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--family", default="first",
            choices=["first", "second", "third", "fourth"])
    parser.add_argument("--alpha", type=float, default=-0.5)
    parser.add_argument("--beta", type=float, default=-0.5)
    args = parser.parse_args()

    plot_padua_nodes(args.alpha, args.beta, args.order, args.family)
