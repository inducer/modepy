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

Transplanted quadrature in one dimension
----------------------------------------

The transplanted maps implemented here include the Hale-Trefethen
conformal-map family and the Kosloff-Tal-Ezer map.

Given a base rule :math:`(s_i, w_i)` on :math:`[-1,1]`, transplanted quadrature
uses a map :math:`x=g(s)` to build

.. math::

    x_i = g(s_i), \qquad \tilde w_i = w_i g'(s_i),

so that

.. math::

    \int_{-1}^1 f(x)\,dx = \int_{-1}^1 f(g(s))g'(s)\,ds
    \approx \sum_i \tilde w_i f(x_i).

Map functions
~~~~~~~~~~~~~

.. currentmodule:: modepy.quadrature.transplanted

Identity map
^^^^^^^^^^^^

Use ``map_name="identity"`` for the unmodified base rule.

.. autofunction:: map_identity

Sausage polynomial maps
^^^^^^^^^^^^^^^^^^^^^^^

Use ``map_name="sausage_d{odd}"`` (for example ``"sausage_d5"``,
``"sausage_d9"``, ``"sausage_d17"``) for odd-degree normalized polynomial
truncations of :math:`\arcsin`.

.. autofunction:: map_sausage

Kosloff-Tal-Ezer map
^^^^^^^^^^^^^^^^^^^^

Use ``map_name="kte"`` (or ``"kosloff_tal_ezer"``).

* ``kte_rho`` (``>1``) sets the default parameterization
  :math:`\alpha = 2/(\rho + \rho^{-1})`.
* ``kte_alpha`` explicitly sets :math:`\alpha` (must satisfy ``0<alpha<1``)
  and overrides ``kte_rho``.

.. autofunction:: map_kosloff_tal_ezer

Strip conformal map
^^^^^^^^^^^^^^^^^^^

Use ``map_name="strip"`` with ``strip_rho > 1``.

.. note::

    The strip map requires interior nodes (``abs(s)<1``), so endpoint rules
    (for example Gauss-Lobatto or Clenshaw-Curtis) are not valid with
    ``map_name="strip"``.

.. autofunction:: map_strip

Map dispatcher
^^^^^^^^^^^^^^

.. autofunction:: map_trefethen_transplant

Quadrature classes
~~~~~~~~~~~~~~~~~~

.. currentmodule:: modepy

Transplanted1DQuadrature
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Transplanted1DQuadrature
    :show-inheritance:

TransplantedLegendreGaussQuadrature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: TransplantedLegendreGaussQuadrature
    :show-inheritance:

Example
~~~~~~~

.. code-block:: python

    import modepy as mp

    q_kte = mp.TransplantedLegendreGaussQuadrature(
        20,
        map_name="kte",
        kte_rho=1.4,
        force_dim_axis=True,
    )

    q_sausage = mp.TransplantedLegendreGaussQuadrature(
        20,
        map_name="sausage_d9",
        force_dim_axis=True,
    )

References
~~~~~~~~~~

* N. Hale and L. N. Trefethen, *New Quadrature Formulas from Conformal
  Maps*, ``SIAM J. Numer. Anal.`` 46(2), 930-948 (2008),
  doi:10.1137/07068607X.
* D. Kosloff and H. Tal-Ezer, *A Modified Chebyshev Pseudospectral Method
  with an O(N^{-1}) Time Step Restriction*,
  ``Journal of Computational Physics`` 104(2), 457-469 (1993),
  doi:10.1006/jcph.1993.1044.

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
