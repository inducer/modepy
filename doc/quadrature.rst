Quadrature
==========

Base classes
------------

.. automodule:: modepy.quadrature

.. currentmodule:: modepy

.. autoclass:: Quadrature

    .. automethod:: __call__

Jacobi-Gauss quadrature in one dimension
----------------------------------------

.. automodule:: modepy.quadrature.jacobi_gauss

.. currentmodule:: modepy

.. autoclass:: Quadrature

.. autoclass:: JacobiGaussQuadrature

.. autoclass:: LegendreGaussQuadrature

.. currentmodule:: modepy.quadrature.jacobi_gauss

.. autofunction:: jacobi_gauss_lobatto_nodes

.. autofunction:: legendre_gauss_lobatto_nodes

Quadratures on the simplex
--------------------------

.. currentmodule:: modepy

.. autoexception:: QuadratureRuleUnavailable

.. autoclass:: GrundmannMoellerSimplexQuadrature

    .. automethod:: __call__

.. autoclass:: XiaoGimbutasSimplexQuadrature

    .. automethod:: __call__

.. autoclass:: VioreanuRokhlinSimplexQuadrature

    .. automethod:: __call__

.. vim: sw=4
