from __future__ import division, absolute_import

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


def _fejer(n):
    r"""Weights of the Fejer2, Clenshaw-Curtis and Fejer1 quadratures
    by DFTs, assuming n >= 2.

    Nodes: x_k = cos(k * pi / n).

    The algorithm follows:
    Jörg Waldvogel,
    Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules,
    BIT Numerical Mathematics, 2003, Vol. 43, No. 1, pp. 001-018,
    https://doi.org/10.1007/s10543-006-0045-4

    """
    N = np.arange(1, n, 2)  # noqa
    l = len(N)  # noqa
    m = n - l
    K = np.arange(0, m)  # noqa

    # Fejer2 nodes: k=0,1,...,n; weights: wf2, wf2_n=wf2_0=0
    v0 = np.concatenate([2 / N / (N - 2), 1 / N[-1:], np.zeros(m)])
    v2 = 0 - v0[:-1] - v0[-1:0:-1]
    wf2 = np.fft.ifft(v2)
    assert np.allclose(wf2.imag, 0)
    wf2 = wf2.real[1:]
    xf2 = np.cos(np.arange(1, n) * np.pi / n)
    print('\nFejer-2:')
    print(xf2)
    print(wf2)
    print(np.sum(wf2) - 2)

    # Clenshaw-Curtis nodes: k=0,1,...,n; weights: wcc, wcc_n=wcc_0
    g0 = -np.ones(n)
    g0[l] = g0[l] + n
    g0[m] = g0[m] + n
    g = g0 / (n**2 - 1 + (n % 2))
    wcc = np.fft.ifft(v2 + g)
    assert np.allclose(wcc.imag, 0)
    wcc = np.concatenate([wcc, wcc[:1]]).real
    xcc = np.cos(np.arange(0, n + 1) * np.pi / n)
    print('\nClenshaw-Curtis:')
    print(xcc)
    print(wcc)
    print(np.sum(wcc) - 2)

    # Fejer1 nodes: k=1/2,3/2,...,n-1/2; vector of weights: wf1
    v0 = np.concatenate(
            [2 * np.exp(1j * np.pi * K / n) / (1 - 4 * (K**2)),
             np.zeros(l+1)])
    v1 = v0[:-1] + np.conj(v0[-1:0:-1])
    wf1 = np.fft.ifft(v1)
    assert np.allclose(wf1.imag, 0)
    wf1 = wf1.real
    xf1 = np.cos((np.arange(0, n) + 0.5) * np.pi / n)
    print('\nFejer-1:')
    print(xf1)
    print(wf1)
    print(np.sum(wf1) - 2)

    # tested against ApproxFun's 10th order rules
    # https://github.com/JuliaApproximation/ApproxFun.jl/issues/316


class ClenshawCurtisQuadrature(Quadrature):
    r"""Clenshaw-Curtis quadrature of order *N*.

    Inherits from :class:`modepy.Quadrature`. See there for the interface
    to obtain nodes and weights.
    """
    pass


if __name__ == "__main__":
    _fejer(9)
