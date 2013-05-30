Interpolation Nodes
===================

Coordinate systems on simplices
-------------------------------

.. _tri-coords:

Coordinates on the triangle
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unit coordinates :math:`(r,s)`::

    C
    |\
    | \
    |  O
    |   \
    |    \
    A-----B

Vertices in unit coordinates::

    O = (0,0)
    A = (-1,-1)
    B = (1,-1)
    C = (-1,1)

Equilateral coordinates :math:`(x,y)`::

          C
         / \
        /   \
       /     \
      /   O   \
     /         \
    A-----------B

Vertices in equilateral coordinates::

    O = (0,0)
    A = (-1,-1/sqrt(3))
    B = (1,-1/sqrt(3))
    C = (0,2/sqrt(3))

.. _tet-coords:

Coordinates on the tetrahedron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unit coordinates :math:`(r,s,t)`::

               ^ s
               |
               C
              /|\
             / | \
            /  |  \
           /   |   \
          /   O|    \
         /   __A-----B---> r
        /_--^ ___--^^
       ,D--^^^
    t L

(squint, and it might start making sense...)

Vertices in unit coordinates::

    O=( 0, 0, 0)
    A=(-1,-1,-1)
    B=(+1,-1,-1)
    C=(-1,+1,-1)
    D=(-1,-1,+1)

Vertices in equilateral coordinates :math:`(x,y,z)`::

    O = (0,0,0)
    A = (-1,-1/sqrt(3),-1/sqrt(6))
    B = ( 1,-1/sqrt(3),-1/sqrt(6))
    C = ( 0, 2/sqrt(3),-1/sqrt(6))
    D = ( 0,         0, 3/sqrt(6))

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
