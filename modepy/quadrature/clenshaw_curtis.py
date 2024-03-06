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

from typing import Tuple

import numpy as np

from modepy.quadrature import Quadrature


def _fejer(n: int, rule: str) -> Tuple[np.ndarray, np.ndarray]:
    r"""Nodes and weights of the Fejer2, Clenshaw-Curtis and Fejer1
    quadratures by DFTs.

    Nodes: x_k = cos(k * pi / n).

    The algorithm follows:

    :arg n: the scheme's order. n is related to the number of quadrature
            nodes as follows:
            - for f1, n_1_nodes = n,     n >= 1
            - for f2, n_q_nodes = n - 1, n >= 2
            - for cc, n_q_nodes = n + 1, n >= 1
              (a minimal cc rule must include the end points)

    :arg rule: any of ``"cc"``, ``"f1"`` or ``"f2"``.
    """
    if not n >= 1:
        raise RuntimeError(f"Invalid n = {n} (must be n >= 1)")

    if rule not in {"f1", "f2", "cc"}:
        raise NotImplementedError(f"Invalide rule: {rule}")

    N = np.arange(1, n, 2)  # noqa: N806
    r = len(N)
    m = n - r
    K = np.arange(0, m)  # noqa: N806

    if rule == "f2" or rule == "cc":

        v0 = np.concatenate([2 / N / (N - 2), 1 / N[-1:], np.zeros(m)])
        v2 = 0 - v0[:-1] - v0[-1:0:-1]

        if rule == "f2":
            # Fejer2 nodes: k=0,1,...,n; weights: wf2, wf2_n=wf2_0=0
            # nodes with zero weights are not included in the return values
            if n == 1:
                raise RuntimeError("Fejer1 does not exist for n=1")
            wf2 = np.fft.ifft(v2)
            assert np.allclose(wf2.imag, 0)
            wf2 = wf2.real[1:]
            xf2 = np.cos(np.arange(1, n) * np.pi / n)
            return xf2, wf2

        if rule == "cc":
            # Clenshaw-Curtis nodes: k=0,1,...,n; weights: wcc, wcc_n=wcc_0
            if n == 1:
                return np.array([-1, 1]), np.array([1, 1])
            g0 = -np.ones(n)
            g0[r] = g0[r] + n
            g0[m] = g0[m] + n
            g = g0 / (n**2 - 1 + (n % 2))
            wcc = np.fft.ifft(v2 + g)
            assert np.allclose(wcc.imag, 0)
            wcc = np.concatenate([wcc, wcc[:1]]).real
            xcc = np.cos(np.arange(0, n + 1) * np.pi / n)
            return xcc, wcc

    if rule == "f1":
        # Fejer1 nodes: k=1/2,3/2,...,n-1/2; vector of weights: wf1
        v0 = np.concatenate(
                [2 * np.exp(1j * np.pi * K / n) / (1 - 4 * (K**2)),
                 np.zeros(r + 1)])
        v1 = v0[:-1] + np.conj(v0[-1:0:-1])
        wf1 = np.fft.ifft(v1)
        assert np.allclose(wf1.imag, 0)
        wf1 = wf1.real
        xf1 = np.cos((np.arange(0, n) + 0.5) * np.pi / n)
        return xf1, wf1

    raise AssertionError


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

        x, w = _fejer(N, "cc")

        if force_dim_axis:
            x = x.reshape(1, -1)

        super().__init__(x, w, exact_to=N)


class FejerQuadrature(Quadrature):
    r"""Fejér quadrature rules of order *N*.

    * Fejér quadrature of the first kind has *N* points and uses Chebyshev
      nodes, i.e. the roots of Chebyshev polynomials.

    * Fejér quadrature of the second kind has *N - 1* points and uses only the
      interior extrema of the Chebyshev nodes, i.e. the true stationary points.
      This rule is alsmost identical to Clenshaw-Curtis and can be nested.

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
            x, w = _fejer(N, "f1")
        elif kind == 2:
            x, w = _fejer(N, "f2")
        else:
            raise ValueError("kind must be either 1 or 2")

        if force_dim_axis:
            x = x.reshape(1, -1)

        super().__init__(x, w)

        self.kind: int = kind
        """Kind of the Fejér quadrature, either first-kind or second-kind."""

# }}}
