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

Gauss-Kronrod quadrature
------------------------

.. automodule:: modepy.quadrature.kronrod

.. _quadrature-transplanted-1d:

Transplanted quadrature in one dimension
----------------------------------------

The transplanted maps implemented here include the conformal maps from
Hale-Trefethen (the sausage polynomial family and the strip map) as well as
the earlier Kosloff-Tal-Ezer :math:`\arcsin` map.

.. note::

    In using the term 'transplanted', we are following the terminology from
    [HaleTrefethen2008]_. In other nomenclature, this is also referred to as a
    change of variables transformation using a conformal mapping.

Given a base rule :math:`(s_i, w_i^{(s)})` on :math:`[-1,1]`, transplanted quadrature
uses a map :math:`g(s): [-1, 1] \to [-1, 1]` to build

.. math::

    x_i = g(s_i), \qquad w_i = w_i^{(s)} g'(s_i),

so that

.. math::

    \int_{-1}^1 f(x)\,\mathrm{d}x = \int_{-1}^1 f(g(s))\,g'(s)\,\mathrm{d}s
    \approx \sum_i w_i f(x_i).

Map functions
~~~~~~~~~~~~~

.. currentmodule:: modepy.quadrature.transplanted

Identity map
^^^^^^^^^^^^

.. autofunction:: map_identity

Sausage polynomial maps
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: map_sausage

Kosloff-Tal-Ezer map
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: map_kosloff_tal_ezer

Strip conformal map
^^^^^^^^^^^^^^^^^^^

.. autofunction:: map_strip

Quadrature wrappers
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: modepy

.. autofunction:: transplanted_1d_quadrature

.. autofunction:: transplanted_legendre_gauss_quadrature

Example
~~~~~~~

.. code-block:: python

    from functools import partial

    import modepy as mp
    from modepy.quadrature.transplanted import map_kosloff_tal_ezer, map_sausage

    q_kte = mp.transplanted_legendre_gauss_quadrature(
        20,
        partial(map_kosloff_tal_ezer, rho=1.4),
        force_dim_axis=True,
    )

    q_sausage = mp.transplanted_legendre_gauss_quadrature(
        20,
        partial(map_sausage, degree=9),
        force_dim_axis=True,
    )

References
~~~~~~~~~~

.. [HaleTrefethen2008] N. Hale and L. N. Trefethen,
    *New Quadrature Formulas from Conformal Maps*,
    *SIAM Journal on Numerical Analysis* 46(2), 930-948 (2008),
    `doi:10.1137/07068607X <https://doi.org/10.1137/07068607X>`__.

.. [KosloffTalEzer1993] D. Kosloff and H. Tal-Ezer,
    *A Modified Chebyshev Pseudospectral Method
    with an* :math:`O(N^{-1})` *Time Step Restriction*,
    *Journal of Computational Physics* 104(2), 457-469 (1993),
    `doi:10.1006/jcph.1993.1044 <https://doi.org/10.1006/jcph.1993.1044>`__.

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
