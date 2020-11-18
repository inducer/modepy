Quadrature
==========

Base classes
------------

.. automodule:: modepy.quadrature

.. currentmodule:: modepy

.. autoclass:: Quadrature

Jacobi-Gauss quadrature in one dimension
----------------------------------------

.. automodule:: modepy.quadrature.jacobi_gauss

.. currentmodule:: modepy

.. autoclass:: JacobiGaussQuadrature

.. autoclass:: LegendreGaussQuadrature

.. autoclass:: ChebyshevGaussQuadrature

.. autoclass:: GaussGegenbauerQuadrature

.. currentmodule:: modepy.quadrature.jacobi_gauss

.. autofunction:: jacobi_gauss_lobatto_nodes

.. autofunction:: legendre_gauss_lobatto_nodes

Clenshaw-Curtis and Fejér quadrature in one dimension
-----------------------------------------------------

.. automodule:: modepy.quadrature.clenshaw_curtis

.. currentmodule:: modepy

.. autoclass:: ClenshawCurtisQuadrature

.. autoclass:: FejerQuadrature

Quadratures on the simplex
--------------------------

.. currentmodule:: modepy

.. autoexception:: QuadratureRuleUnavailable

.. autoclass:: GrundmannMoellerSimplexQuadrature

.. autoclass:: XiaoGimbutasSimplexQuadrature

.. autoclass:: VioreanuRokhlinSimplexQuadrature


Quadratures on the hypercube
----------------------------

.. currentmodule:: modepy

.. autoclass:: WitherdenVincentQuadrature

.. autoclass:: TensorProductQuadrature
.. autoclass:: LegendreGaussTensorProductQuadrature

.. vim: sw=4
