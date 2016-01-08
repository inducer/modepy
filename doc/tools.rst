Tools
=====

Version query
-------------

.. module:: modepy.version

.. data:: VERSION

    A tuple like *(2013,1,1)* indicating modepy's version.

.. data:: VERSION_TEXT

    A textual representation of the modepy version.

Interpolation matrices
----------------------

.. currentmodule:: modepy

.. autofunction:: vandermonde

Vandermonde matrices are very useful to concisely express interpolation. For
instance, one may use the inverse :math:`V^{-1}` of a Vandermonde matrix
:math:`V` to map nodal values (i.e. point values at the nodes corresponding to
:math:`V`) into modal (i.e. basis) coefficients. :math:`V` itself maps modal
coefficients to nodal values. This allows easy resampling:

.. autofunction:: resampling_matrix

Vandermonde matrices also obey the following relationship useful for obtaining
point interpolants:

.. math::

    V^T [\text{interpolation coefficents to point $x$}] = \phi_i(x),

where :math:`(\phi_i)_i` is the basis of functions underlying :math:`V`.

.. autofunction:: inverse_mass_matrix

.. autofunction:: mass_matrix

.. autofunction:: modal_face_mass_matrix

.. autofunction:: nodal_face_mass_matrix

Differentiation is also convenient to express by using :math:`V^{-1}` to
obtain modal values and then using a Vandermonde matrix for the derivatives
of the basis to return to nodal values.

.. autofunction:: differentiation_matrices

Modal decay/residual
--------------------

.. automodule:: modepy.modal_decay

.. autofunction:: fit_modal_decay
.. autofunction:: estimate_relative_expansion_residual

Interpolation quality
---------------------

.. currentmodule:: modepy.tools

.. autofunction:: estimate_lebesgue_constant

.. vim: sw=4
