modepy: Basis Functions, Node Sets, Quadratures
===============================================

.. image:: https://gitlab.tiker.net/inducer/modepy/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/modepy/commits/main
.. image:: https://github.com/inducer/modepy/actions/workflows/ci.yml/badge.svg
    :alt: Github Build Status
    :target: https://github.com/inducer/modepy/actions/workflows/ci.yml
.. image:: https://badge.fury.io/py/modepy.png
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/modepy/
.. image:: https://zenodo.org/badge/9846038.svg
    :alt: Zenodo DOI for latest release
    :target: https://zenodo.org/doi/10.5281/zenodo.11105051

``modepy`` helps you create well-behaved high-order discretizations on
simplices (i.e. segments, triangles and tetrahedra) and tensor products of
simplices (i.e. squares, cubes, prisms, etc.). These are a key building block
for high-order unstructured discretizations, as often used in a finite
element context. Features include:

- Support for simplex and tensor product elements in any dimension.
- Orthogonal bases:
    - Jacobi polynomials with derivatives
    - Orthogonal polynomials for simplices up to 3D and tensor product elements
      and their derivatives.
    - All bases permit symbolic evaluation, for code generation.
- Access to numerous quadrature rules:
    - Jacobi-Gauss, Jacobi-Gauss-Lobatto in 1D
      (includes Legendre, Chebyshev, ultraspherical, Gegenbauer)
    - Clenshaw-Curtis and Fejér in 1D
    - Grundmann-Möller on the simplex
    - Xiao-Gimbutas on the simplex
    - Vioreanu-Rokhlin on the simplex
    - Jaśkowiec-Sukumar on the tetrahedron
    - Witherden-Vincent on the hypercube
    - Generic tensor products built on the above, e.g. for prisms and hypercubes
- Matrices for FEM, usable across all element types:
    - generalized Vandermonde,
    - mass matrices (including lumped diagonal),
    - face mass matrices,
    - differentiation matrices, and
    - resampling matrices.
- Objects to represent 'element shape' and 'function space',
  generic node/mode/quadrature retrieval based on them.

Its roots closely followed the approach taken in the book

  Hesthaven, Jan S., and Tim Warburton. "Nodal Discontinuous Galerkin Methods:
  Algorithms, Analysis, and Applications". 1st ed. Springer, 2007.
  `Book web page <http://nudg.org>`_

but much has been added beyond that basic functionality.

Resources:

* `documentation <http://documen.tician.de/modepy>`_
* `wiki home page <http://wiki.tiker.net/ModePy>`_
* `source code via git <http://github.com/inducer/modepy>`_
