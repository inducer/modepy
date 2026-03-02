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

from functools import lru_cache
from importlib import import_module
from math import asin, isfinite, log, sqrt
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from modepy.quadrature import Quadrature


if TYPE_CHECKING:
    from collections.abc import Callable

    from modepy.typing import ArrayF


class _RootScalarResult(Protocol):
    converged: bool
    root: float


class _RootScalarFn(Protocol):
    def __call__(
        self,
        f: Callable[[float], float],
        *,
        bracket: tuple[float, float],
        method: str,
    ) -> _RootScalarResult: ...


class _EllipkFn(Protocol):
    def __call__(self, x: float) -> float: ...


class _EllipjFn(Protocol):
    def __call__(
        self, u: ArrayF, m: float
    ) -> tuple[ArrayF, ArrayF, ArrayF, ArrayF]: ...


def _scipy_attr(module_name: str, attr_name: str) -> object:
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            "The Trefethen strip map requires scipy. "
            "Install modepy with scipy support to use map_name='strip'."
        ) from exc

    try:
        return cast("object", getattr(module, attr_name))
    except AttributeError as exc:
        raise RuntimeError(
            f"scipy module '{module_name}' is missing required attribute '{attr_name}'"
        ) from exc


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
    denom = np.asarray(sum(coeffs), dtype=s.dtype)

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

    Reference:
        D. Kosloff and H. Tal-Ezer, "A Modified Chebyshev Pseudospectral
        Method with an O(N^{-1}) Time Step Restriction," Journal of
        Computational Physics 104(2), 457-469 (1993),
        doi:10.1006/jcph.1993.1044.
    """
    if alpha is None:
        if rho <= 1.0:
            raise ValueError(f"KTE parameter rho must be > 1: {rho}")

        alpha = float(2.0 / (rho + 1.0 / rho))

    if not (0.0 < alpha < 1.0) or not isfinite(alpha):
        raise ValueError(f"KTE parameter alpha must satisfy 0 < alpha < 1: {alpha}")

    alpha = float(alpha)
    denom = asin(alpha)

    g = np.asarray(np.arcsin(alpha * s) / denom, dtype=np.float64)
    gp = np.asarray(
        alpha / (denom * np.sqrt(1.0 - alpha**2 * s**2)),
        dtype=np.float64,
    )

    return g, gp


def _map_preserves_exact_to(map_name: str, *, sausage_degree: int) -> bool:
    if map_name == "identity":
        return True

    legacy_sausage_degree = _sausage_degree_from_map_name(map_name)
    if legacy_sausage_degree is not None:
        return legacy_sausage_degree == 1

    return map_name == "sausage" and sausage_degree == 1


def _sausage_degree_from_map_name(map_name: str) -> int | None:
    if not map_name.startswith("sausage_d"):
        return None

    degree_text = map_name[len("sausage_d") :]
    if not degree_text.isdigit():
        raise ValueError(
            f"unsupported sausage map '{map_name}'. Expected format: sausage_d{{odd}}"
        )

    return int(degree_text)


@lru_cache(maxsize=16)
def _strip_map_parameter_m(rho: float) -> float:
    if rho <= 1.0:
        raise ValueError(f"strip map parameter rho must be > 1: {rho}")

    root_scalar = cast("_RootScalarFn", _scipy_attr("scipy.optimize", "root_scalar"))
    ellipk = cast("_EllipkFn", _scipy_attr("scipy.special", "ellipk"))

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

    ellipj = cast("_EllipjFn", _scipy_attr("scipy.special", "ellipj"))
    ellipk = cast("_EllipkFn", _scipy_attr("scipy.special", "ellipk"))

    m = _strip_map_parameter_m(rho)
    m4 = sqrt(sqrt(m))
    k = float(ellipk(m))

    omega = 2.0 * k * np.arcsin(s) / np.pi
    sn_jacobi, cn_jacobi, dn_jacobi, _ = ellipj(omega, m)
    sn = np.asarray(sn_jacobi, dtype=np.float64)
    cn = np.asarray(cn_jacobi, dtype=np.float64)
    dn = np.asarray(dn_jacobi, dtype=np.float64)

    g = np.asarray(np.arctanh(m4 * sn) / np.arctanh(m4), dtype=np.float64)
    gp = np.asarray(
        (
            2.0
            * k
            * m4
            * cn
            * dn
            / (
                np.pi
                * np.sqrt(1.0 - s**2)
                * (1.0 - np.sqrt(m) * sn**2)
                * np.arctanh(m4)
            )
        ),
        dtype=np.float64,
    )

    return g, gp


def map_trefethen_transplant(
    s: ArrayF,
    map_name: str,
    *,
    sausage_degree: int = 9,
    strip_rho: float = 1.4,
    kte_rho: float = 1.4,
    kte_alpha: float | None = None,
) -> tuple[ArrayF, ArrayF]:
    """Map 1D nodes to a Trefethen transplanted quadrature rule.

    :arg s: nodes on :math:`[-1, 1]`.
    :arg map_name: one of ``identity``, ``sausage``, ``kte``,
        ``kosloff_tal_ezer``, ``strip``.
    :arg sausage_degree: odd polynomial degree for ``map_name="sausage"``.
    :arg strip_rho: strip-map parameter for ``map_name="strip"``.
    :arg kte_rho: KTE parameter for ``map_name in {"kte", "kosloff_tal_ezer"}``
        when ``kte_alpha`` is not supplied.
    :arg kte_alpha: explicit KTE :math:`\alpha` override.

    :returns: ``(mapped_nodes, jacobian)``.

    The supported maps are:

    * ``identity``: :func:`map_identity`
    * ``sausage``: :func:`map_sausage`
    * ``sausage_d{odd}`` (legacy alias): :func:`map_sausage`
    * ``kte`` / ``kosloff_tal_ezer``: :func:`map_kosloff_tal_ezer`
    * ``strip``: :func:`map_strip`

    """
    if map_name == "identity":
        return map_identity(s)

    if map_name == "sausage":
        return map_sausage(s, sausage_degree)

    legacy_sausage_degree = _sausage_degree_from_map_name(map_name)
    if legacy_sausage_degree is not None:
        return map_sausage(s, legacy_sausage_degree)

    if map_name == "strip":
        return map_strip(s, rho=strip_rho)

    if map_name in {"kte", "kosloff_tal_ezer"}:
        return map_kosloff_tal_ezer(s, rho=kte_rho, alpha=kte_alpha)

    raise ValueError(
        "unsupported map_name "
        f"'{map_name}'. Expected one of: "
        "identity, sausage, sausage_d{odd}, kte, kosloff_tal_ezer, strip"
    )


def transplanted_1d_quadrature(
    quadrature: Quadrature,
    map_name: str = "sausage",
    *,
    sausage_degree: int = 9,
    strip_rho: float = 1.4,
    kte_rho: float = 1.4,
    kte_alpha: float | None = None,
) -> Quadrature:
    r"""Map an existing 1D quadrature rule using a Trefethen transplant map.

    The transformed rule approximates

    .. math::

        \int_{-1}^1 f(x)\,dx = \int_{-1}^1 f(g(s)) g'(s)\,ds,

    by mapping existing nodes :math:`s_i` and scaling existing weights :math:`w_i`
    with :math:`g'(s_i)`.
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

    mapped_nodes, jacobian = map_trefethen_transplant(
        nodes_1d,
        map_name=map_name,
        sausage_degree=sausage_degree,
        strip_rho=strip_rho,
        kte_rho=kte_rho,
        kte_alpha=kte_alpha,
    )
    mapped_weights = quadrature.weights * jacobian

    if force_dim_axis:
        mapped_nodes = np.reshape(mapped_nodes, (1, mapped_nodes.shape[0]))

    exact_to = None
    if _map_preserves_exact_to(map_name, sausage_degree=sausage_degree):
        try:
            exact_to = quadrature.exact_to
        except AttributeError:
            exact_to = None

    return Quadrature(mapped_nodes, mapped_weights, exact_to=exact_to)


def transplanted_legendre_gauss_quadrature(
    n: int,
    map_name: str = "sausage",
    *,
    sausage_degree: int = 9,
    strip_rho: float = 1.4,
    kte_rho: float = 1.4,
    kte_alpha: float | None = None,
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
        map_name=map_name,
        sausage_degree=sausage_degree,
        strip_rho=strip_rho,
        kte_rho=kte_rho,
        kte_alpha=kte_alpha,
    )
