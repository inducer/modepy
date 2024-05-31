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


from typing import Callable, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
import numpy.linalg as la

from modepy.modes import Basis, BasisNotOrthonormal
from modepy.quadrature import Quadrature
from modepy.shapes import Face


__doc__ = r"""
.. currentmodule:: modepy

.. autofunction:: vandermonde

.. autofunction:: multi_vandermonde

Vandermonde matrices are very useful to concisely express interpolation. For
instance, one may use the inverse :math:`V^{-1}` of a Vandermonde matrix
:math:`V` to map nodal values (i.e. point values at the nodes corresponding to
:math:`V`) into modal (i.e. basis) coefficients. :math:`V` itself maps modal
coefficients to nodal values. This allows easy resampling:

.. autofunction:: resampling_matrix

Vandermonde matrices also obey the following relationship useful for obtaining
point interpolants:

.. math::

    V^T [\text{interpolation coefficients to point $x$}] = \phi_i(x),

where :math:`(\phi_i)_i` is the basis of functions underlying :math:`V`.

.. autofunction:: inverse_mass_matrix
.. autofunction:: mass_matrix
.. autofunction:: modal_mass_matrix_for_face
.. autofunction:: nodal_mass_matrix_for_face
.. autofunction:: nodal_quad_mass_matrix_for_face

Differentiation is also convenient to express by using :math:`V^{-1}` to
obtain modal values and then using a Vandermonde matrix for the derivatives
of the basis to return to nodal values.

.. autofunction:: differentiation_matrices
.. autofunction:: diff_matrices

.. autofunction:: diff_matrix_permutation
"""


def vandermonde(
            functions: Sequence[Callable[[np.ndarray], np.ndarray]],
            nodes: np.ndarray
        ) -> np.ndarray:
    """Return a (generalized) Vandermonde matrix.

    The Vandermonde Matrix is given by :math:`V_{i,j} := f_j(x_i)`
    where *functions* is the list of :math:`f_j` and nodes is
    the array of :math:`x_i`, shaped as *(d, npts)*, where *d*
    is the number of dimensions and *npts* is the number of nodes.
    """

    nnodes = nodes.shape[-1]
    nfunctions = len(functions)

    if not functions:
        return np.empty((nnodes, nfunctions), nodes.dtype)

    f_iter = iter(functions)
    f_first = next(f_iter)

    f_first_values = f_first(nodes)

    if isinstance(f_first_values, tuple):
        warn("Calling vandermonde on tuple-returning functions is deprecated. "
             "This will stop working in 2025. "
             "Call multi_vandermonde instead.",
             DeprecationWarning, stacklevel=2)
        return multi_vandermonde(functions, nodes)

    result = np.empty((nnodes, nfunctions), f_first_values.dtype)
    result[:, 0] = f_first_values

    for j, f in enumerate(f_iter):
        result[:, j + 1] = f(nodes)

    return result


def multi_vandermonde(
            functions: Sequence[Callable[[np.ndarray], Sequence[np.ndarray]]],
            nodes: np.ndarray
        ) -> Tuple[np.ndarray, ...]:
    """Evaluate multiple (generalized) Vandermonde matrices.

    The Vandermonde Matrix is given by :math:`V_{i,j} := f_j(x_i)`
    where *functions* is the list of :math:`f_j` and nodes is
    the array of :math:`x_i`, shaped as *(d, npts)*, where *d*
    is the number of dimensions and *npts* is the number of nodes.
    *functions* must return :class:`tuple` instances.
    A sequence of matrices is returned--i.e. this function
    works directly on :func:`modepy.Basis.gradients` and returns
    a tuple of matrices.
    """

    nnodes = nodes.shape[-1]
    nfunctions = len(functions)

    if not nfunctions:
        raise ValueError("empty functions is not allowed")

    result = None
    for j, f in enumerate(functions):
        f_values = f(nodes)

        if result is None:
            from pytools import single_valued
            dtype = single_valued(fi.dtype for fi in f_values)
            result = tuple(np.empty((nnodes, nfunctions), dtype)
                    for i in range(len(f_values)))

        for i, f_values_i in enumerate(f_values):
            result[i][:, j] = f_values_i

    assert result is not None

    return result


def resampling_matrix(
            basis: Sequence[Callable[[np.ndarray], np.ndarray]],
            new_nodes: np.ndarray,
            old_nodes: np.ndarray,
            least_squares_ok: bool = False
        ) -> np.ndarray:
    """Return a matrix that maps nodal values on *old_nodes* onto nodal
    values on *new_nodes*.

    :arg basis: A sequence of basis functions accepting
        arrays of shape *(dims, npts)*, like those returned by
        :func:`modepy.orthonormal_basis_for_space`.
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
            raise RuntimeError(
                    f"number of input nodes ({old_nodes.shape[1]}) "
                    f"and number of basis functions ({len(basis)}) "
                    "do not agree--perhaps use least_squares_ok")


def differentiation_matrices(
            basis: Sequence[Callable[[np.ndarray], np.ndarray]],
            grad_basis: Sequence[Callable[[np.ndarray], Sequence[np.ndarray]]],
            nodes: np.ndarray,
            from_nodes: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, ...]:
    """Return matrices carrying out differentiation on nodal values in the
    :math:`(r,s,t)` unit directions. (See :ref:`tri-coords` and
    :ref:`tet-coords`.)

    :arg basis: A sequence of basis functions accepting arrays
        of shape *(dims, npts)*,
        like those returned by :func:`modepy.Basis.functions`.
    :arg grad_basis: A sequence of functions returning the
        gradients of *basis*,
        like those in :attr:`modepy.Basis.gradients`.
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

    if len(basis) != from_nodes.shape[-1]:
        raise ValueError("basis is not unisolvent, cannot interpolate")

    vdm = vandermonde(basis, from_nodes)
    grad_vdms = multi_vandermonde(grad_basis, nodes)

    return tuple(
            np.asarray(la.solve(vdm.T, gv.T).T, order="C")
            for gv in grad_vdms)


def diff_matrices(
            basis: Basis,
            nodes: np.ndarray,
            from_nodes: Optional[np.ndarray] = None
        ):
    """Like :func:`differentiation_matrices`, but for a given :class:`~modepy.Basis`.

    .. versionadded :: 2024.2
    """
    return differentiation_matrices(
                   basis.functions, basis.gradients, nodes, from_nodes)


def diff_matrix_permutation(
            node_tuples: Sequence[Tuple[int, ...]],
            ref_axis: int
        ) -> np.ndarray:
    """Return a :mod:`numpy` array *permutation* of integers so that::

        diff_matrices[ref_axis] == diff_matrices[0][permutation][:, permutation]

    .. versionadded:: 2020.1
    """
    ntup_index_lookup = {nt: i for i, nt in enumerate(node_tuples)}

    if ref_axis == 0:
        return np.arange(len(node_tuples), dtype=np.intp)

    permutation = np.zeros(len(node_tuples), dtype=np.intp)
    for i, nt in enumerate(node_tuples):
        swapped = list(nt)
        swapped[0], swapped[ref_axis] = swapped[ref_axis], swapped[0]
        swapped_t = tuple(swapped)
        flipped_idx = ntup_index_lookup[swapped_t]
        permutation[i] = flipped_idx

    return permutation


def inverse_mass_matrix(
            basis: Union[Basis, Sequence[Callable[[np.ndarray], np.ndarray]]],
            nodes: np.ndarray
        ) -> np.ndarray:
    """Return a matrix :math:`A=M^{-1}`, which is the inverse of the one returned
    by :func:`mass_matrix`. Requires that the basis is orthonormal with weight 1.

    .. versionadded:: 2015.1
    """

    if isinstance(basis, Basis):
        try:
            if basis.orthonormality_weight() != 1:
                raise NotImplementedError(
                    "inverse mass matrix of non-orthogonal basis")
        except BasisNotOrthonormal:
            raise NotImplementedError("inverse mass matrix of non-orthogonal basis")
        basis_functions: Sequence[Callable[[np.ndarray], np.ndarray]] = \
            basis.functions
    else:
        basis_functions = basis

        from warnings import warn
        warn("Passing a sequence of functions to inverse_mass_matrix is deprecated "
             "and will stop working in 2025. Pass a Basis instead.",
             DeprecationWarning, stacklevel=2)

    vdm = vandermonde(basis_functions, nodes)

    return np.dot(vdm, vdm.T)


def mass_matrix(
            basis: Union[Basis, Sequence[Callable[[np.ndarray], np.ndarray]]],
            nodes: np.ndarray
        ) -> np.ndarray:
    r"""Return a mass matrix :math:`M`, which obeys

    .. math::

        M_{ij} = \int_\triangle \phi_i(x) \phi_j(x) dx = (V^{-T} V^{-1})_{ij}.

    :arg basis: assumed to be an orthonormal basis with respect to the :math:`L^2`
        inner product.

    .. versionadded:: 2014.1
    """

    return la.inv(inverse_mass_matrix(basis, nodes))


def modal_mass_matrix_for_face(
            face: Face, face_quad: Quadrature,
            trial_functions: Sequence[Callable[[np.ndarray], np.ndarray]],
            test_functions: Sequence[Callable[[np.ndarray], np.ndarray]]
        ) -> np.ndarray:
    r"""Using the quadrature *face_quad*, provide a matrix :math:`M_f` that
    satisfies:

    .. math::

        \displaystyle {(M_f)}_{ij} \approx \int_F \phi_i(r) \psi_j(r) dr,

    where :math:`\phi_i` are the (volume) basis functions
    *test_functions*, and :math:`\psi_j` are the (surface)
    *trial_functions*.

    .. versionadded:: 2020.3
    """

    mapped_nodes = face.map_to_volume(face_quad.nodes)

    result = np.empty((len(test_functions), len(trial_functions)))

    for i, test_f in enumerate(test_functions):
        test_vals = test_f(mapped_nodes)
        for j, trial_f in enumerate(trial_functions):
            result[i, j] = (test_vals*trial_f(face_quad.nodes)) @ face_quad.weights

    return result


def nodal_mass_matrix_for_face(
            face: Face, face_quad: Quadrature,
            trial_functions: Sequence[Callable[[np.ndarray], np.ndarray]],
            test_functions: Sequence[Callable[[np.ndarray], np.ndarray]],
            volume_nodes: np.ndarray, face_nodes: np.ndarray
        ) -> np.ndarray:
    r"""Using the quadrature *face_quad*, provide a matrix :math:`M_f` that
    satisfies:

    .. math::

        \displaystyle {(M_f)}_{ij} \approx \int_F \phi_i(r) \psi_j(r) dr,

    where :math:`\phi_i` are the (volume) Lagrange basis functions obtained from
    *test_functions* at *volume_nodes*, and :math:`\psi_j` are the (surface)
    Lagrange basis functions obtained from *trial_functions* at *face_nodes*.

    .. versionadded :: 2020.3
    """
    face_vdm = vandermonde(trial_functions, face_nodes)
    vol_vdm = vandermonde(test_functions, volume_nodes)

    nface_nodes, nface_funcs = face_vdm.shape
    if nface_nodes != nface_funcs:
        raise ValueError(
                "nodal_mass_matrix_for_face received a different number of facial "
                "nodes than facial trial_functions.")
    else:
        face_vdm_inv = la.inv(face_vdm)

    modal_fmm = modal_mass_matrix_for_face(
            face, face_quad, trial_functions, test_functions)
    return la.inv(vol_vdm.T).dot(modal_fmm).dot(face_vdm_inv)


def nodal_quad_mass_matrix_for_face(
            face: Face, face_quad: Quadrature,
            test_functions: Sequence[Callable[[np.ndarray], np.ndarray]],
            volume_nodes: np.ndarray,
        ) -> np.ndarray:
    r"""Using the quadrature *face_quad*, provide a matrix :math:`M_f` that
    satisfies:

    .. math::

        \displaystyle (M_f \boldsymbol u)_i = \sum_j w_j \phi_i(r_j) u_j,

    where :math:`\phi_i` are the (volume) Lagrange basis functions obtained from
    *test_functions* at *volume_nodes*, :math:`w_j` and :math:`r_j` are the
    weights and nodes from *face_quad*, and :math:`u_j` are the point
    values of the trial function at the nodes of *face_quad*.

    .. versionadded :: 2021.2
    """
    vol_vdm = vandermonde(test_functions, volume_nodes)

    mapped_nodes = face.map_to_volume(face_quad.nodes)

    vol_modal_mass_matrix = np.empty((len(test_functions), len(face_quad.weights)))

    for i, test_f in enumerate(test_functions):
        vol_modal_mass_matrix[i] = test_f(mapped_nodes) * face_quad.weights

    return la.inv(vol_vdm.T) @ vol_modal_mass_matrix


# vim: foldmethod=marker
