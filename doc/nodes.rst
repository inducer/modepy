Interpolation Nodes
===================

Simplices
^^^^^^^^^

Transformations between coordinate systems
------------------------------------------

.. currentmodule:: modepy.tools

All of these expect and return arrays of shape *(dims, npts)*.

.. autofunction:: equilateral_to_unit
.. autofunction:: barycentric_to_unit
.. autofunction:: unit_to_barycentric
.. autofunction:: barycentric_to_equilateral

Node sets for interpolation
---------------------------

.. currentmodule:: modepy

.. autofunction:: equidistant_nodes
.. autofunction:: warp_and_blend_nodes

Also see :class:`modepy.VioreanuRokhlinSimplexQuadrature` if nodes on the
boundary are not required.

Hypercubes
^^^^^^^^^^

Node sets for interpolation
---------------------------

.. currentmodule:: modepy

.. autofunction:: tensor_product_nodes
.. autofunction:: legendre_gauss_lobatto_tensor_product_nodes
