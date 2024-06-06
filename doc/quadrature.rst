Quadrature
==========

Base classes
------------

.. automodule:: modepy.quadrature

Jacobi-Gauss quadrature in one dimension
----------------------------------------

.. automodule:: modepy.quadrature.jacobi_gauss

Clenshaw-Curtis and Fejér quadrature in one dimension
-----------------------------------------------------

.. automodule:: modepy.quadrature.clenshaw_curtis

.. currentmodule:: modepy

.. autoclass:: ClenshawCurtisQuadrature
    :show-inheritance:

.. autoclass:: FejerQuadrature
    :members:
    :show-inheritance:

Quadratures on the simplex
--------------------------

.. currentmodule:: modepy

.. autoexception:: QuadratureRuleUnavailable

.. autoclass:: GrundmannMoellerSimplexQuadrature
    :members:
    :show-inheritance:

.. autoclass:: XiaoGimbutasSimplexQuadrature
    :members:
    :show-inheritance:

.. autoclass:: VioreanuRokhlinSimplexQuadrature
    :members:
    :show-inheritance:

.. autoclass::  JaskowiecSukumarQuadrature
    :members:
    :show-inheritance:

Quadratures on the hypercube
----------------------------

.. currentmodule:: modepy

.. autoclass:: WitherdenVincentQuadrature
    :members:
    :show-inheritance:

.. autoclass:: TensorProductQuadrature
    :show-inheritance:
.. autoclass:: LegendreGaussTensorProductQuadrature
    :show-inheritance:

.. vim: sw=4
