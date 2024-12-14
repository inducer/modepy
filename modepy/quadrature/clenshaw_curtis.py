from __future__ import annotations


__copyright__ = "Copyright (C) 2019 Xiaoyu Wei"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np

from modepy.quadrature import Quadrature


# {{{ Clenshaw-Curtis

def _make_clenshaw_curtis_nodes_and_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the Clenshaw-Curtis quadrature."""
    if n < 1:
        raise ValueError(f"Clenshaw-Curtis order must be at least 1: n = {n}")

    if n == 1:
        return np.array([-1, 1]), np.array([1, 1])

    N = np.arange(1, n, 2)  # noqa: N806
    r = len(N)
    m = n - r

    # Clenshaw-Curtis nodes
    x = np.cos(np.arange(0, n + 1) * np.pi / n)

    # Clenshaw-Curtis weights
    w = np.concatenate([2 / N / (N - 2), 1 / N[-1:], np.zeros(m)])
    w = 0 - w[:-1] - w[-1:0:-1]
    g0: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = -np.ones(n)
    g0[r] = g0[r] + n
    g0[m] = g0[m] + n
    g0 = g0 / (n**2 - 1 + (n % 2))
    w = np.fft.ifft(w + g0)
    assert np.allclose(w.imag, 0)

    wr = w.real
    return x, np.concatenate([wr, wr[:1]])


class ClenshawCurtisQuadrature(Quadrature):
    r"""Clenshaw-Curtis quadrature of order *N* with *N + 1* points.

    The quadrature rule is exact up to degree :math:`N` and can be nested.
    Its performance for differentiable functions is comparable with the classic
    Gauss-Legendre quadrature, which is exact for polynomials of degree up
    to :math:`2N + 1`. Implementation is based on [Waldvogel2003]_.

    Integrates on the interval :math:`(-1, 1)`.

    .. [Waldvogel2003] Jörg Waldvogel,
        *Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules*,
        BIT Numerical Mathematics, 2003, Vol. 43, No. 1, pp. 001-018.
        `DOI <https://doi.org/10.1007/s10543-006-0045-4>`__
    """

    def __init__(self, N: int, force_dim_axis: bool = False) -> None:  # noqa: N803
        if not force_dim_axis:
            from warnings import warn
            warn("setting 'force_dim_axis' to 'False' is deprecated and "
                    "makes 1d rules inconsistent with higher dimensions. "
                    "This option will go away in 2022",
                    DeprecationWarning, stacklevel=2)

        x, w = _make_clenshaw_curtis_nodes_and_weights(N)

        if force_dim_axis:
            x = x.reshape(1, -1)

        super().__init__(x, w, exact_to=N)

# }}}


# {{{ Fejer

def _make_fejer1_nodes_and_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the Fejer quadrature of the first kind."""
    if n < 1:
        raise ValueError(f"Fejer1 order must be at least 1: n = {n}")

    N = np.arange(1, n, 2)  # noqa: N806
    r = len(N)
    m = n - r
    K = np.arange(0, m)  # noqa: N806

    # Fejer1 nodes: k = 1/2, 3/2, ..., n-1/2
    x = np.cos((np.arange(0, n) + 0.5) * np.pi / n)

    # Fejer1 weights
    w = np.concatenate([
        2 * np.exp(1j * np.pi * K / n) / (1 - 4 * (K**2)), np.zeros(r + 1)
        ])
    w = w[:-1] + np.conj(w[-1:0:-1])
    w = np.fft.ifft(w)

    assert np.allclose(w.imag, 0)
    return x, w.real


def _make_fejer2_nodes_and_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the Fejer quadrature of the second kind."""
    if n < 2:
        raise ValueError(f"Fejer2 order must be at least 2: n = {n}")

    N = np.arange(1, n, 2)  # noqa: N806
    r = len(N)
    m = n - r

    # Fejer2 nodes: k=0, 1, ..., n
    x = np.cos(np.arange(1, n) * np.pi / n)

    # Fejer2 weights
    w = np.concatenate([2 / N / (N - 2), 1 / N[-1:], np.zeros(m)])
    w = 0 - w[:-1] - w[-1:0:-1]
    w = np.fft.ifft(w)[1:]

    assert np.allclose(w.imag, 0)
    return x, w.real


class FejerQuadrature(Quadrature):
    r"""Fejér quadrature rules of order *N*.

    * Fejér quadrature of the first kind has *N* points and uses Chebyshev
      nodes, i.e. the roots of Chebyshev polynomials.

    * Fejér quadrature of the second kind has *N - 1* points and uses only the
      interior extrema of the Chebyshev nodes, i.e. the true stationary points.
      This rule is almost identical to Clenshaw-Curtis and can be nested.

    Integrates on the interval :math:`(-1, 1)`. Implementation is based on
    [Waldvogel2003]_.
    """

    def __init__(self,
                 N: int,  # noqa: N803
                 kind: int = 1,
                 force_dim_axis: bool = False) -> None:
        if not force_dim_axis:
            from warnings import warn
            warn("setting 'force_dim_axis' to 'False' is deprecated and "
                    "makes 1d rules inconsistent with higher dimensions. "
                    "This option will go away in 2022",
                    DeprecationWarning, stacklevel=2)

        if kind == 1:
            x, w = _make_fejer1_nodes_and_weights(N)
        elif kind == 2:
            x, w = _make_fejer2_nodes_and_weights(N)
        else:
            raise ValueError("kind must be either 1 or 2")

        if force_dim_axis:
            x = x.reshape(1, -1)

        super().__init__(x, w)

        self.kind: int = kind
        """Kind of the Fejér quadrature, either first-kind or second-kind."""

# }}}
