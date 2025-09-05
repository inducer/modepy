Welcome to modepy's documentation!
==================================

.. module:: modepy

:mod:`modepy` helps you create well-behaved high-order discretizations on
simplices (i.e. triangles and tetrahedra) and tensor products of simplices
(i.e. squares, cubes, prisms, etc.). These are a key building block for
high-order unstructured discretizations, as often used in a finite element
context.

The basic objects that :mod:`modepy` manipulates are functions on a shape (or
reference domain). For example, it supplies an orthonormal basis on triangles
(shown below).

.. image:: images/pkdo-2d.png
    :width: 100%
    :align: center
    :alt: Proriol-Koornwinder-Dubiner orthogonal (PKDO) basis functions of order 3

The file that created this plot is included in the :mod:`modepy` distribution
as :download:`examples/plot-basis.py <../examples/plot-basis.py>`.

Its roots closely followed the approach taken in the Hesthaven and Warburton
book [HesthavenWarburton2007]_, but much has been added beyond that basic
functionality.

.. [HesthavenWarburton2007] J. S. Hesthaven and T. Warburton (2007).
    *Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications*
    (1st ed.).
    `doi:10.1007/978-0-387-72067-8 <https://doi.org/10.1007/978-0-387-72067-8>`__.

Example
-------

Here's an idea of code that uses :mod:`modepy`:

.. literalinclude:: ../examples/prism-forms.py
    :language: python
    :linenos:
    :lines: 3-
    :lineno-start: 3
    :caption: Example code that constructs a prism domain and computes a weak derivative on it.

This file is included in the :mod:`modepy` distribution as
:download:`examples/prism-forms.py <../examples/prism-forms.py>`.

modepy around the web
---------------------

* `Documentation <https://documen.tician.de/modepy>`__.
* `Source code <https://github.com/inducer/modepy>`__.
* `Bug tracker <https://github.com/inducer/modepy/issues>`__.
* `PyPI <https://pypi.org/project/modepy/>`__.

Contents
========

.. toctree::
    :maxdepth: 2

    shapes
    modes
    nodes
    quadrature
    quad_construction
    tools
    misc
    🚀 Github <https://github.com/inducer/modepy>
    💾 Download Releases <https://pypi.org/project/modepy>

* :ref:`genindex`
* :ref:`modindex`

.. vim: sw=4
