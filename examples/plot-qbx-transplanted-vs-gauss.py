from __future__ import annotations

# Optional QBX dependencies (meshmode/pytential/sumpy) are not installed in CI.
# pyright: basic, reportMissingImports=false
import tempfile
import warnings
from functools import lru_cache, partial
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import meshmode.mesh.generation as mgen
import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools
from arraycontext import flatten
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
    PolynomialGivenNodesElementGroup,
)
from pytential import GeometryCollection, bind, sym
from pytential.array_context import PyOpenCLArrayContext
from pytential.qbx import QBXLayerPotentialSource
from scipy.optimize import root_scalar
from scipy.special import ellipk
from sumpy.expansion.local import LineTaylorLocalExpansion
from sumpy.kernel import LaplaceKernel
from sumpy.qbx import LayerPotential

import modepy as mp
from modepy.quadrature import Transformed1DQuadrature
from modepy.quadrature.transplanted import (
    map_kosloff_tal_ezer,
    map_sausage,
    map_strip,
)


NPANELS, MODE = 8, 8
QBX_ORDER = 20
ASSOC_TOL = 0.05
NPTS = list(range(4, 30))
# Each entry is (label, map_fn_factory) where factory takes strip_rho and returns
# a bound map function (or None for plain Gauss-Legendre).
MAPS = [
    ("gauss",     lambda rho: None),
    ("kte",       lambda rho: partial(map_kosloff_tal_ezer, rho=rho)),
    ("strip",     lambda rho: partial(map_strip, rho=rho)),
    ("sausage_d5", lambda rho: partial(map_sausage, degree=5)),
    ("sausage_d9", lambda rho: partial(map_sausage, degree=9)),
]
OUT = Path(tempfile.gettempdir()) / "qbx-transplanted-vs-gauss-2d.png"

STRIP_RHO: float | None = None
STRIP_SAFETY = 0.5


@lru_cache(maxsize=16)
def strip_map_parameter_m(rho: float) -> float:
    target = 4.0 * np.log(rho) / np.pi

    def f(m: float) -> float:
        return float(ellipk(1.0 - m) / ellipk(m) - target)

    upper = 1.0 - 1.0e-8
    while f(upper) > 0.0 and 1.0 - upper > 1.0e-16:
        upper = 1.0 - (1.0 - upper) / 10.0
    result = root_scalar(f, bracket=(1.0e-14, upper), method="brentq")
    if not result.converged:
        raise RuntimeError("failed to solve strip-map parameter m")
    return float(result.root)


def strip_half_width(rho: float) -> float:
    return float(np.pi / (4.0 * np.arctanh(strip_map_parameter_m(rho) ** 0.25)))


def strip_rho_for_half_width(
    target_half_width: float, rho_min: float = 1.05, rho_max: float = 5.0
) -> float:
    if target_half_width <= strip_half_width(rho_min):
        return float(rho_min)
    if target_half_width >= strip_half_width(rho_max):
        return float(rho_max)

    def f(rho: float) -> float:
        return strip_half_width(rho) - target_half_width

    return float(root_scalar(f, bracket=(rho_min, rho_max), method="brentq").root)


def kte_alpha_for_rho(rho: float) -> float:
    return float(2.0 / (rho + 1.0 / rho))


def make_quad(
    npts: int,
    map_fn,
) -> mp.Quadrature:
    if map_fn is None:
        return mp.LegendreGaussQuadrature(npts - 1, force_dim_axis=True)
    return mp.transplanted_legendre_gauss_quadrature(
        npts - 1,
        map_fn,
        force_dim_axis=True,
    )


def make_group(order: int, quad: mp.Quadrature):
    class _G(PolynomialGivenNodesElementGroup):
        def __init__(self, meg):
            super().__init__(meg, order, quad.nodes)

        def quadrature_rule(self):
            return quad

        def discretization_key(self):
            return (
                type(self),
                self.dim,
                self.order,
                tuple(quad.nodes.ravel()),
                tuple(quad.weights.ravel()),
            )

    return _G


def make_mesh_and_t(panel_edges: np.ndarray, npts: int, unit_nodes: np.ndarray):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Unimplemented: Cannot check element orientation for a mesh with "
                "mesh.dim != mesh.ambient_dim"
            ),
            category=UserWarning,
        )
        return mgen.make_curve_mesh(
            mgen.circle,
            panel_edges,
            order=npts - 1,
            unit_nodes=unit_nodes,
            node_vertex_consistency_tolerance=False,
            return_parametrization_points=True,
        )


def source_ds_weights(quad: mp.Quadrature, panel_edges: np.ndarray) -> np.ndarray:
    dtw = np.concatenate([
        Transformed1DQuadrature(quad, a, b).weights for a, b in pairwise(panel_edges)
    ])
    return (2.0 * np.pi) * dtw


def gauss_centers_radii(actx, panel_edges: np.ndarray, npts: int):
    qg = make_quad(npts, None)
    mesh, _ = make_mesh_and_t(panel_edges, npts, qg.nodes)
    discr = Discretization(actx, mesh, make_group(npts - 1, qg))
    qbx = QBXLayerPotentialSource(
        discr,
        fine_order=1,
        qbx_order=1,
        fmm_order=False,
        target_association_tolerance=ASSOC_TOL,
    )
    places = GeometryCollection(qbx)
    centers = actx.to_numpy(
        flatten(bind(places, sym.expansion_centers(2, +1))(actx), actx)
    ).reshape(2, -1)
    radii = actx.to_numpy(flatten(bind(places, sym.expansion_radii(2))(actx), actx))
    return centers, radii


def auto_strip_rho(actx, panel_edges: np.ndarray) -> float:
    _, radii = gauss_centers_radii(actx, panel_edges, max(NPTS))
    eta_min = float(np.min(radii) / (np.pi / NPANELS))
    target_half_width = STRIP_SAFETY * eta_min
    rho = strip_rho_for_half_width(target_half_width)
    print(
        "auto strip rho: "
        f"eta_min={eta_min:.6f}, "
        f"target_half_width={target_half_width:.6f}, "
        f"rho={rho:.4f}"
    )
    return rho


def eval_rule(
    actx,
    lpot,
    panel_edges: np.ndarray,
    npts: int,
    map_fn,
    targets: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    quad = make_quad(npts, map_fn)
    mesh, t_src = make_mesh_and_t(panel_edges, npts, quad.nodes)
    sources = mesh.groups[0].nodes.reshape(2, -1)
    sigma = np.cos(MODE * 2.0 * np.pi * t_src)
    strengths = (actx.from_numpy(sigma * source_ds_weights(quad, panel_edges)),)
    (result,) = lpot(
        actx,
        actx.from_numpy(targets),
        actx.from_numpy(sources),
        actx.from_numpy(centers),
        strengths,
        expansion_radii=actx.from_numpy(radii),
    )
    return actx.to_numpy(result)


def main() -> None:
    panel_edges = np.linspace(0.0, 1.0, NPANELS + 1)
    queue = cl.CommandQueue(cl.create_some_context(interactive=False))
    allocator = cl_tools.ImmediateAllocator(queue)
    actx = PyOpenCLArrayContext(queue, allocator=allocator)
    strip_rho = (
        STRIP_RHO if STRIP_RHO is not None else auto_strip_rho(actx, panel_edges)
    )
    print(
        "using strip "
        f"rho={strip_rho:.4f} "
        f"(half-width={strip_half_width(strip_rho):.6f})"
    )

    lknl = LaplaceKernel(2)
    lpot = LayerPotential(
        expansion=LineTaylorLocalExpansion(lknl, QBX_ORDER),
        target_kernels=(lknl,),
        source_kernels=(lknl,),
    )

    maps = [(name, factory(strip_rho)) for name, factory in MAPS]
    names = [name for name, _ in maps]

    orders, totals = [], []
    errors = {name: [] for name in names}

    print("QBX convergence on meshmode circle (frozen Gauss targets+centers)")
    print("order  total_nodes  " + "  ".join(f"{n:>10s}" for n in names))
    print("-" * (33 + 12 * len(names)))

    for i, npts in enumerate(NPTS):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=cl.CompilerWarning)

            qg = make_quad(npts, None)
            tgt_mesh, t_tgt = make_mesh_and_t(panel_edges, npts, qg.nodes)
            targets = tgt_mesh.groups[0].nodes.reshape(2, -1)
            centers, radii = gauss_centers_radii(actx, panel_edges, npts)
            ref = np.cos(MODE * 2.0 * np.pi * t_tgt) / (2.0 * MODE)

            orders.append(npts - 1)
            totals.append(NPANELS * npts)
            for name, map_fn in maps:
                values = eval_rule(
                    actx,
                    lpot,
                    panel_edges,
                    npts,
                    map_fn,
                    targets,
                    centers,
                    radii,
                )
                errors[name].append(float(np.max(np.abs(values - ref))))

        vals = "  ".join(f"{errors[name][i]:10.3e}" for name in names)
        print(f"{orders[i]:5d}  {totals[i]:11d}  {vals}")

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    style = {
        "gauss": ("o-", "Gauss-Legendre"),
        "kte": (
            "D-",
            f"KTE (rho={strip_rho:.3f}, alpha={kte_alpha_for_rho(strip_rho):.3f})",
        ),
        "strip": ("s-", f"Strip (rho={strip_rho:.3f})"),
        "sausage_d5": ("^-", "Sausage d5"),
        "sausage_d9": ("v-", "Sausage d9"),
    }
    for name in names:
        marker, label = style[name]
        ax.semilogy(orders, errors[name], marker, label=label)
    ax.set_xlabel("per-panel quadrature order")
    ax.set_ylabel("max abs error vs circle eigenvalue")
    ax.set_title(
        f"2D direct QBX on circle, frozen centers, mode={MODE}, qbx_order={QBX_ORDER}"
    )
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best")
    fig.savefig(OUT, dpi=160)
    print(f"Saved plot to: {OUT}")


if __name__ == "__main__":
    main()
