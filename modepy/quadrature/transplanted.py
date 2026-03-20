from __future__ import annotations


r"""
.. currentmodule:: modepy.quadrature.transplanted

Transplanted quadrature applies a smooth map :math:`x=g(s)` to an existing
one-dimensional rule on :math:`[-1,1]`.

Given base nodes/weights :math:`(s_i, w_i^{(s)})`, the transplanted rule is

.. math::

    x_i = g(s_i), \qquad \tilde w_i = w_i^{(s)} g'(s_i),

so that

.. math::

    \int_{-1}^1 f(x)\,dx = \int_{-1}^1 f(g(s)) g'(s)\,ds
    \approx \sum_i \tilde w_i f(x_i).

For map names, parameters, examples, and references, see
:ref:`quadrature-transplanted-1d`.
"""

from functools import lru_cache, partial
from math import asin, isfinite, log, sqrt
from typing import TYPE_CHECKING, cast

import numpy as np

from modepy.quadrature import Quadrature


if TYPE_CHECKING:
    from collections.abc import Callable

    from modepy.typing import ArrayF


def map_identity(s: ArrayF) -> tuple[ArrayF, ArrayF]:
    """Identity transplant map on :math:`[-1, 1]`.

    Returns ``(s, 1)``.
    """
    return np.array(s, copy=True), np.ones_like(s)


def _arcsin_taylor_coefficients(max_odd_degree: int) -> tuple[float, ...]:
    if max_odd_degree < 1 or max_odd_degree % 2 == 0:
        raise ValueError(
            f"sausage must use positive odd degree, got: {max_odd_degree}"
        )

    nterms = (max_odd_degree + 1) // 2
    coeffs = [1.0]

    for k in range(1, nterms):
        km1 = k - 1
        coeffs.append(
            coeffs[km1] * (2.0 * km1 + 1.0) ** 2 / (2.0 * k * (2.0 * km1 + 3.0))
        )

    return tuple(coeffs)


def map_sausage(s: ArrayF, degree: int) -> tuple[ArrayF, ArrayF]:
    r"""Odd-degree polynomial sausage map from Hale-Trefethen (2008).

    This is the normalized odd Taylor truncation of :math:`\arcsin(s)`
    through the monomial of degree *degree*.

    :arg degree: positive odd degree in ``{1, 3, 5, ...}``.
    """
    coeffs = _arcsin_taylor_coefficients(degree)
    denom = sum(coeffs)

    g = np.zeros_like(s)
    gp = np.zeros_like(s)

    for k, c_k in enumerate(coeffs):
        power = 2 * k + 1
        g = g + c_k * s**power
        gp = gp + c_k * power * s ** (power - 1)

    return g / denom, gp / denom


def map_kosloff_tal_ezer(
    s: ArrayF,
    *,
    rho: float = 1.4,
    alpha: float | None = None,
) -> tuple[ArrayF, ArrayF]:
    r"""Kosloff-Tal-Ezer map.

    The map is

    .. math::

        g(s) = \frac{\arcsin(\alpha s)}{\arcsin(\alpha)},

    where :math:`0 < \alpha < 1`.

    If *alpha* is not provided, it is chosen from *rho* using

    .. math::

        \alpha = \frac{2}{\rho + \rho^{-1}},

    matching the parameter choice discussed by Hale-Trefethen for a
    :math:`\rho`-ellipse analyticity model.

    .. rubric:: Reference

    [KosloffTalEzer1993]_
    """
    if alpha is None:
        if rho <= 1.0:
            raise ValueError(f"KTE parameter rho must be > 1: {rho}")

        alpha = float(2.0 / (rho + 1.0 / rho))

    if not (0.0 < alpha < 1.0) or not isfinite(alpha):
        raise ValueError(f"KTE parameter alpha must satisfy 0 < alpha < 1: {alpha}")

    alpha = float(alpha)
    denom = asin(alpha)

    g = np.arcsin(alpha * s) / denom
    gp = alpha / (denom * np.sqrt(1.0 - alpha**2 * s**2))

    return g, gp



@lru_cache(maxsize=16)
def _strip_map_parameter_m(rho: float) -> float:
    if rho <= 1.0:
        raise ValueError(f"strip map parameter rho must be > 1: {rho}")

    try:
        from scipy.optimize import root_scalar
        from scipy.special import ellipk
    except ImportError as exc:
        raise RuntimeError(
            "The Trefethen strip map requires scipy. "
            "Install modepy with scipy support to use map_name='strip'."
        ) from exc

    target = 4.0 * log(rho) / np.pi

    def f(m: float) -> float:
        return float(ellipk(1.0 - m) / ellipk(m) - target)

    upper = 1.0 - 1.0e-8
    while f(upper) > 0.0 and 1.0 - upper > 1.0e-16:
        upper = 1.0 - (1.0 - upper) / 10.0

    if f(upper) > 0.0:
        raise RuntimeError(f"failed to bracket strip-map parameter for rho={rho}")

    result = root_scalar(f, bracket=(1.0e-14, upper), method="brentq")
    if not result.converged:
        raise RuntimeError("failed to solve strip-map parameter m")

    return float(result.root)


def map_strip(s: ArrayF, *, rho: float = 1.4) -> tuple[ArrayF, ArrayF]:
    r"""Strip map from Hale-Trefethen transplanted quadrature.

    :arg rho: strip parameter, must satisfy ``rho > 1``.

    .. important::

        This map requires interior nodes (``abs(s) < 1``), so it is intended
        for base rules such as Legendre-Gauss that do not include endpoints.
    """
    if np.any(np.abs(s) >= 1.0):
        raise ValueError("strip map expects interior nodes, i.e. abs(s) < 1")

    try:
        from scipy.special import ellipj, ellipk
    except ImportError as exc:
        raise RuntimeError(
            "The Trefethen strip map requires scipy. "
            "Install modepy with scipy support to use map_name='strip'."
        ) from exc

    m = _strip_map_parameter_m(rho)
    m4 = sqrt(sqrt(m))
    k = float(ellipk(m))

    omega = 2.0 * k * np.arcsin(s) / np.pi
    sn_jacobi, cn_jacobi, dn_jacobi, _ = ellipj(omega, m)

    g = np.arctanh(m4 * sn_jacobi) / np.arctanh(m4)
    gp = (
            2.0
            * k
            * m4
            * cn_jacobi
            * dn_jacobi
            / (
                np.pi
                * np.sqrt(1.0 - s**2)
                * (1.0 - np.sqrt(m) * sn_jacobi**2)
                * np.arctanh(m4)
            )
        )

    return g, gp


def transplanted_1d_quadrature(
    quadrature: Quadrature,
    map_fn: Callable[[ArrayF], tuple[ArrayF, ArrayF]],
) -> Quadrature:
    r"""Map an existing 1D quadrature rule using a Trefethen transplant map.

    The transformed rule approximates

    .. math::

        \int_{-1}^1 f(x)\,dx = \int_{-1}^1 f(g(s)) g'(s)\,ds,

    by mapping existing nodes :math:`s_i` and scaling existing weights :math:`w_i`
    with :math:`g'(s_i)`.

    :arg map_fn: a callable ``(s: ArrayF) -> (nodes, jacobian)``, such as
        :func:`~modepy.quadrature.transplanted.map_identity`,
        :func:`~modepy.quadrature.transplanted.map_sausage`,
        :func:`~modepy.quadrature.transplanted.map_kosloff_tal_ezer`,
        or :func:`~modepy.quadrature.transplanted.map_strip`,
        with parameters (if any) bound via :func:`functools.partial`.
    """
    base_nodes = quadrature.nodes
    if base_nodes.ndim == 1:
        nodes_1d = np.asarray(base_nodes)
        force_dim_axis = False
    elif base_nodes.ndim == 2 and base_nodes.shape[0] == 1:
        nodes_1d = cast("ArrayF", base_nodes[0])
        force_dim_axis = True
    else:
        raise ValueError(
            "transplanted_1d_quadrature requires a one-dimensional base quadrature"
        )

    mapped_nodes, jacobian = map_fn(nodes_1d)
    mapped_weights = quadrature.weights * jacobian

    if force_dim_axis:
        mapped_nodes = np.reshape(mapped_nodes, (1, mapped_nodes.shape[0]))

    exact_to = None
    if map_fn is map_identity:
        try:
            exact_to = quadrature.exact_to
        except AttributeError:
            exact_to = None

    return Quadrature(mapped_nodes, mapped_weights, exact_to=exact_to)


def transplanted_legendre_gauss_quadrature(
    n: int,
    map_fn: Callable[[ArrayF], tuple[ArrayF, ArrayF]],
    *,
    backend: str | None = None,
    force_dim_axis: bool = False,
) -> Quadrature:
    r"""Legendre-Gauss quadrature transplanted by a Trefethen map."""
    from modepy.quadrature.jacobi_gauss import LegendreGaussQuadrature

    return transplanted_1d_quadrature(
        LegendreGaussQuadrature(
            n,
            backend=backend,
            force_dim_axis=force_dim_axis,
        ),
        map_fn,
    )
