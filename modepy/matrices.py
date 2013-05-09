from __future__ import division

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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




def vandermonde(functions, points):
    """Return a Vandermonde matrix.

    The Vandermonde Matrix is given by :math:`V_{i,j} := f_j(x_i)`
    where *functions* is the list of :math:`f_j` and points is
    the array of :math:`x_i`, shaped as *(d, npts)*, where *d*
    is the number of dimensions and *npts* is the number of points.

    *functions* are allowed to return :class:`tuple` instances.
    """

    npoints = points.shape[-1]
    nfunctions = len(functions)

    result = None
    for j, f in enumerate(functions):
        f_values = f(points)

        if result is None:
            if isinstance(f_values, tuple):
                from pytools import single_valued
                dtype = single_valued(fi.dtype for fi in f_values)
                result = [np.empty((npoints, nfunctions), dtype)
                        for i in range(len(f_values))]
            else:
                result = np.empty((npoints, nfunctions), f_values.dtype)

        if isinstance(f_values, tuple):
            for i, f_values_i in enumerate(f_values):
                result[i][:, j] = f_values_i
        else:
            result[:, j] = f_values

    return result

def mass_matrix(basis, nodes):
    pass

def differentiation_matrix(basis, nodes):
    pass

# vim: foldmethod=marker
