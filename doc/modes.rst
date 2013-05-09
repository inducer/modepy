Modes (Basis functions)
=======================

.. automodule:: modepy.modes

Jacobi polynomials
------------------

.. autofunction:: jacobi

.. autofunction:: grad_jacobi

Dimension-independent basis getters
-----------------------------------

.. |proriol-ref| replace::
    Proriol, Joseph. "Sur une famille de polynomes á deux variables orthogonaux
    dans un triangle." CR Acad. Sci. Paris 245 (1957): 2459-2461.

.. |koornwinder-ref| replace::
    Koornwinder, T. "Two-variable analogues of the classical orthogonal polynomials."
    Theory and Applications of Special Functions. 1975, pp. 435-495.

.. |dubiner-ref| replace::
    Dubiner, Moshe. "Spectral Methods on Triangles and Other Domains." Journal of
    Scientific Computing 6, no. 4 (December 1, 1991): 345–390.
    http://dx.doi.org/10.1007/BF01060030

.. autofunction:: get_simplex_onb

.. autofunction:: get_grad_simplex_onb

Dimension-specific functions
----------------------------

.. autofunction:: pkdo_2d

.. autofunction:: grad_pkdo_2d

.. autofunction:: pkdo_3d

.. autofunction:: grad_pkdo_3d


.. vim: sw=4
