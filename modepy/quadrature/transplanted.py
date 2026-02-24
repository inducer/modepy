from __future__ import annotations


r"""
.. currentmodule:: modepy.quadrature.transplanted

Transplanted quadrature applies a smooth map :math:`x=g(s)` to an existing
one-dimensional rule on :math:`[-1,1]`.

Given base nodes/weights :math:`(s_i, w_i)`, the transplanted rule is

.. math::

    x_i = g(s_i), \qquad \tilde w_i = w_i g'(s_i),

so that

.. math::

    \int_{-1}^1 f(x)\,dx = \int_{-1}^1 f(g(s)) g'(s)\,ds
    \approx \sum_i \tilde w_i f(x_i).

The map dispatcher :func:`map_trefethen_transplant` recognizes these map names:

* ``"identity"``
* ``"sausage_d{odd}"`` (for example ``"sausage_d5"``, ``"sausage_d9"``,
  ``"sausage_d17"``)
* ``"kte"`` or ``"kosloff_tal_ezer"``
* ``"strip"``

Parameter notes:

* ``strip_rho`` controls the strip-map conformal parameter, with ``strip_rho > 1``.
* ``kte_rho`` controls the default KTE parameterization through
  :math:`\alpha = 2 / (\rho + \rho^{-1})`, with ``kte_rho > 1``.
* ``kte_alpha`` explicitly sets :math:`\alpha` (must satisfy ``0 < kte_alpha < 1``)
  and overrides ``kte_rho``.

.. note::

    The strip map requires interior nodes (``abs(s) < 1``). Endpoint-including
    base rules (for example Gauss-Lobatto or Clenshaw-Curtis) are therefore not
    valid with ``map_name="strip"``.

.. autofunction:: map_identity
.. autofunction:: map_sausage
.. autofunction:: map_kosloff_tal_ezer
.. autofunction:: map_strip
.. autofunction:: map_trefethen_transplant

.. currentmodule:: modepy

.. autoclass:: Transplanted1DQuadrature
.. autoclass:: TransplantedLegendreGaussQuadrature
"""

from functools import lru_cache
from math import isfinite
from typing import TYPE_CHECKING, Any

import numpy as np

from modepy.quadrature import Quadrature


if TYPE_CHECKING:
    from modepy.typing import ArrayF


def map_identity(s: ArrayF) -> tuple[ArrayF, ArrayF]:
    """Identity transplant map on :math:`[-1, 1]`.

    Returns ``(s, 1)``.
    """
    return s.copy(), np.ones_like(s)


def _arcsin_taylor_coefficients(max_odd_degree: int) -> ArrayF:
    if max_odd_degree < 1 or max_odd_degree % 2 == 0:
        raise ValueError(f"sausage degree must be positive and odd: {max_odd_degree}")

    nterms = (max_odd_degree + 1) // 2
    coeffs = np.empty(nterms, dtype=np.float64)
    coeffs[0] = 1.0

    for k in range(1, nterms):
        km1 = k - 1
        coeffs[k] = coeffs[km1] * (2.0 * km1 + 1.0) ** 2 / (2.0 * k * (2.0 * km1 + 3.0))

    return coeffs


def map_sausage(s: ArrayF, degree: int) -> tuple[ArrayF, ArrayF]:
    r"""Odd-degree polynomial sausage map from Hale-Trefethen (2008).

    This is the normalized odd Taylor truncation of :math:`\arcsin(s)`
    through the monomial of degree *degree*.

    :arg degree: positive odd degree in ``{1, 3, 5, ...}``.
    """
    coeffs = _arcsin_taylor_coefficients(degree)
    denom = float(np.sum(coeffs))

    g = np.zeros_like(s)
    gp = np.zeros_like(s)

    for k, c_k in enumerate(coeffs):
        power = 2 * k + 1
        g = g + c_k * s**power
        gp = gp + c_k * power * s ** (power - 1)

    return g / denom, gp / denom


def _default_kte_alpha(rho: float) -> float:
    if rho <= 1.0:
        raise ValueError(f"KTE parameter rho must be > 1: {rho}")

    return float(2.0 / (rho + 1.0 / rho))


def _kte_alpha(*, rho: float, alpha: float | None) -> float:
    if alpha is None:
        alpha = _default_kte_alpha(rho)

    if not (0.0 < alpha < 1.0) or not isfinite(alpha):
        raise ValueError(f"KTE parameter alpha must satisfy 0 < alpha < 1: {alpha}")

    return float(alpha)


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
    alpha = _kte_alpha(rho=rho, alpha=alpha)
    denom = np.arcsin(alpha)

    g = np.arcsin(alpha * s) / denom
    gp = alpha / (denom * np.sqrt(1.0 - alpha**2 * s**2))

    return g, gp


def _sausage_degree_from_map_name(map_name: str) -> int | None:
    if not map_name.startswith("sausage_d"):
        return None

    degree_text = map_name[len("sausage_d") :]
    if not degree_text.isdigit():
        raise ValueError(
            f"unsupported sausage map '{map_name}'. Expected format: sausage_d{{odd}}"
        )

    degree = int(degree_text)
    if degree < 1 or degree % 2 == 0 or not isfinite(degree):
        raise ValueError(
            f"unsupported sausage degree in '{map_name}'. "
            "Expected a positive odd degree, e.g. sausage_d5"
        )

    return degree


def _require_scipy_for_strip_map() -> tuple[Any, Any, Any]:
    try:
        from scipy.optimize import root_scalar
        from scipy.special import ellipj, ellipk
    except ImportError as exc:
        raise RuntimeError(
            "The Trefethen strip map requires scipy. "
            "Install modepy with scipy support to use map_name='strip'."
        ) from exc

    return root_scalar, ellipj, ellipk


@lru_cache(maxsize=16)
def _strip_map_parameter_m(rho: float) -> float:
    if rho <= 1.0:
        raise ValueError(f"strip map parameter rho must be > 1: {rho}")

    root_scalar, _, ellipk = _require_scipy_for_strip_map()

    target = 4.0 * np.log(rho) / np.pi

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

    _, ellipj, ellipk = _require_scipy_for_strip_map()

    m = _strip_map_parameter_m(rho)
    m4 = m**0.25
    k = float(ellipk(m))

    omega = 2.0 * k * np.arcsin(s) / np.pi
    sn, cn, dn, _ = ellipj(omega, m)

    g = np.arctanh(m4 * sn) / np.arctanh(m4)
    gp = (
        2.0
        * k
        * m4
        * cn
        * dn
        / (np.pi * np.sqrt(1.0 - s**2) * (1.0 - np.sqrt(m) * sn**2) * np.arctanh(m4))
    )

    return g, gp


def map_trefethen_transplant(
    s: ArrayF,
    map_name: str,
    *,
    strip_rho: float = 1.4,
    kte_rho: float = 1.4,
    kte_alpha: float | None = None,
) -> tuple[ArrayF, ArrayF]:
    """Map 1D nodes to a Trefethen transplanted quadrature rule.

    :arg s: nodes on :math:`[-1, 1]`.
    :arg map_name: one of ``identity``, ``sausage_d{odd}``, ``kte``,
        ``kosloff_tal_ezer``, ``strip``.
    :arg strip_rho: strip-map parameter for ``map_name="strip"``.
    :arg kte_rho: KTE parameter for ``map_name in {"kte", "kosloff_tal_ezer"}``
        when ``kte_alpha`` is not supplied.
    :arg kte_alpha: explicit KTE :math:`\alpha` override.

    :returns: ``(mapped_nodes, jacobian)``.

    The supported maps are:

    * ``identity``: :func:`map_identity`
    * ``sausage_d{odd}``: :func:`map_sausage`
    * ``kte`` / ``kosloff_tal_ezer``: :func:`map_kosloff_tal_ezer`
    * ``strip``: :func:`map_strip`

    Reference:
        N. Hale and L. N. Trefethen, "New Quadrature Formulas from
        Conformal Maps," SIAM Journal on Numerical Analysis 46(2),
        930-948 (2008),
        doi:10.1137/07068607X.

        D. Kosloff and H. Tal-Ezer, "A Modified Chebyshev Pseudospectral
        Method with an O(N^{-1}) Time Step Restriction," Journal of
        Computational Physics 104(2), 457-469 (1993),
        doi:10.1006/jcph.1993.1044.
    """
    if map_name == "identity":
        return map_identity(s)

    sausage_degree = _sausage_degree_from_map_name(map_name)
    if sausage_degree is not None:
        return map_sausage(s, sausage_degree)

    if map_name == "strip":
        return map_strip(s, rho=strip_rho)

    if map_name in {"kte", "kosloff_tal_ezer"}:
        return map_kosloff_tal_ezer(s, rho=kte_rho, alpha=kte_alpha)

    raise ValueError(
        "unsupported map_name "
        f"'{map_name}'. Expected one of: "
        "identity, sausage_d{odd}, kte, kosloff_tal_ezer, strip"
    )


class Transplanted1DQuadrature(Quadrature):
    r"""Map an existing 1D quadrature rule using a Trefethen transplant map.

    The transformed rule approximates

    .. math::

        \int_{-1}^1 f(x)\,dx = \int_{-1}^1 f(g(s)) g'(s)\,ds,

    by mapping existing nodes :math:`s_i` and scaling existing weights :math:`w_i`
    with :math:`g'(s_i)`.

    Reference:
        N. Hale and L. N. Trefethen, "New Quadrature Formulas from
        Conformal Maps," SIAM Journal on Numerical Analysis 46(2),
        930-948 (2008),
        doi:10.1137/07068607X.

        D. Kosloff and H. Tal-Ezer, "A Modified Chebyshev Pseudospectral
        Method with an O(N^{-1}) Time Step Restriction," Journal of
        Computational Physics 104(2), 457-469 (1993),
        doi:10.1006/jcph.1993.1044.
    """

    def __init__(
        self,
        quadrature: Quadrature,
        map_name: str = "sausage_d9",
        *,
        strip_rho: float = 1.4,
        kte_rho: float = 1.4,
        kte_alpha: float | None = None,
    ) -> None:
        base_nodes = quadrature.nodes
        if base_nodes.ndim == 1:
            nodes_1d = base_nodes
            force_dim_axis = False
        elif base_nodes.ndim == 2 and base_nodes.shape[0] == 1:
            nodes_1d = base_nodes[0]
            force_dim_axis = True
        else:
            raise ValueError(
                "Transplanted1DQuadrature requires a one-dimensional base quadrature"
            )

        mapped_nodes, jacobian = map_trefethen_transplant(
            nodes_1d,
            map_name=map_name,
            strip_rho=strip_rho,
            kte_rho=kte_rho,
            kte_alpha=kte_alpha,
        )
        mapped_weights = quadrature.weights * jacobian

        if force_dim_axis:
            mapped_nodes = mapped_nodes.reshape(1, -1)

        super().__init__(mapped_nodes, mapped_weights)

        self.base_quadrature = quadrature
        self.map_name = map_name
        self.strip_rho = strip_rho
        self.kte_rho = kte_rho
        self.kte_alpha = kte_alpha


class TransplantedLegendreGaussQuadrature(Transplanted1DQuadrature):
    r"""Legendre-Gauss quadrature transplanted by a Trefethen map."""

    def __init__(
        self,
        N: int,  # noqa: N803
        map_name: str = "sausage_d9",
        *,
        strip_rho: float = 1.4,
        kte_rho: float = 1.4,
        kte_alpha: float | None = None,
        backend: str | None = None,
        force_dim_axis: bool = False,
    ) -> None:
        from modepy.quadrature.jacobi_gauss import LegendreGaussQuadrature

        super().__init__(
            LegendreGaussQuadrature(
                N,
                backend=backend,
                force_dim_axis=force_dim_axis,
            ),
            map_name=map_name,
            strip_rho=strip_rho,
            kte_rho=kte_rho,
            kte_alpha=kte_alpha,
        )
