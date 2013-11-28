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
import numpy.linalg as la


def vandermonde(functions, nodes):
    """Return a (generalized) Vandermonde matrix.

    The Vandermonde Matrix is given by :math:`V_{i,j} := f_j(x_i)`
    where *functions* is the list of :math:`f_j` and nodes is
    the array of :math:`x_i`, shaped as *(d, npts)*, where *d*
    is the number of dimensions and *npts* is the number of nodes.

    *functions* are allowed to return :class:`tuple` instances.
    In this case, a tuple of matrices is returned--i.e. this function
    works directly on :func:`modepy.grad_simplex_onb` and returns
    a tuple of matrices.
    """

    nnodes = nodes.shape[-1]
    nfunctions = len(functions)

    result = None
    for j, f in enumerate(functions):
        f_values = f(nodes)

        if result is None:
            if isinstance(f_values, tuple):
                from pytools import single_valued
                dtype = single_valued(fi.dtype for fi in f_values)
                result = tuple(np.empty((nnodes, nfunctions), dtype)
                        for i in range(len(f_values)))
            else:
                result = np.empty((nnodes, nfunctions), f_values.dtype)

        if isinstance(f_values, tuple):
            for i, f_values_i in enumerate(f_values):
                result[i][:, j] = f_values_i
        else:
            result[:, j] = f_values

    return result


def resampling_matrix(basis, new_nodes, old_nodes, least_squares_ok=False):
    """Return a matrix that maps nodal values on *old_nodes* onto nodal
    values on *new_nodes*.

    :arg basis: A sequence of basis functions accepting
        arrays of shape *(dims, npts)*, like those returned by
        :func:`modepy.simplex_onb`.
    :arg new_nodes: An array of shape *(dims, n_new_nodes)*
    :arg old_nodes: An array of shape *(dims, n_old_nodes)*
    :arg least_squares_ok: If *False*, then nodal values at *old_nodes*
        are required to determine the interpolant uniquely, i.e. the
        Vandermonde matrix must be square. If *True*, then a
        point-wise
        `least-squares best-approximant
        <http://en.wikipedia.org/wiki/Least_squares>`_
        is used (by ways of the
        `pseudo-inverse
        <https://en.wikipedia.org/wiki/Moore-Penrose_pseudoinverse>`_
        of the Vandermonde matrix).
    """

    vdm_old = vandermonde(basis, old_nodes)
    vdm_new = vandermonde(basis, new_nodes)

    # Hooray for efficiency. :)
    n_modes_in = len(basis)
    n_modes_out = vdm_new.shape[1]
    resample = np.eye(max(n_modes_in, n_modes_out))[:n_modes_out, :n_modes_in]

    resample_vdm_new = np.dot(vdm_new, resample)

    def is_square(m):
        return m.shape[0] == m.shape[1]

    if is_square(vdm_old):
        return np.asarray(
                la.solve(vdm_old.T, resample_vdm_new.T).T,
                order="C")
    else:
        if least_squares_ok:
            return np.asarray(
                    np.dot(resample_vdm_new, la.pinv(vdm_old)),
                    order="C")
        else:
            raise RuntimeError("number of input nodes and number "
                    "of basis functions "
                    "do not agree--perhaps use least_squares_ok")


def differentiation_matrices(basis, grad_basis, nodes, from_nodes=None):
    """Return matrices carrying out differentiation on nodal values in the
    :math:`(r,s,t)` unit directions. (See :ref:`tri-coords` and
    :ref:`tet-coords`.)

    :arg basis: A sequence of basis functions accepting arrays
        of shape *(dims, npts)*,
        like those returned by :func:`modepy.simplex_onb`.
    :arg grad_basis: A sequence of functions returning the
        gradients of *basis*,
        like those returned by :func:`modepy.grad_simplex_onb`.
    :arg nodes: An array of shape *(dims, n_nodes)*
    :arg from_nodes:  An array of shape *(dims, n_from_nodes)*.
        If *None*, assumed to be the same as *nodes*.
    :returns: If *grad_basis* returned tuples (i.e. in 2D and 3D), returns
        a tuple of length *dims* containing differentiation matrices.
        If not, returns just one differentiation matrix.

    .. versionchanged:: 2013.4

        Added *from_nodes*.
    """
    if from_nodes is None:
        from_nodes = nodes

    vdm = vandermonde(basis, from_nodes)
    grad_vdms = vandermonde(grad_basis, nodes)

    if isinstance(grad_vdms, tuple):
        return tuple(
                np.asarray(la.solve(vdm.T, gv.T).T, order="C")
                for gv in grad_vdms)
    else:
        return np.asarray(
                la.solve(vdm.T, grad_vdms.T).T,
                order="C")


# vim: foldmethod=marker
