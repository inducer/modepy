Shapes
======

`modepy.shapes` provides a generic description of the supported shapes
(i.e. reference elements).

Interface
^^^^^^^^^

.. currentmodule:: modepy.shapes

.. autoclass:: Shape
.. autofunction:: get_unit_vertices
.. autofunction:: get_face_vertex_indices
.. autofunction:: get_face_map

Simplices
^^^^^^^^^

.. autoclass:: Simplex

.. _tri-coords:

Coordinates on the triangle
---------------------------

Unit coordinates :math:`(r, s)`::

    ^ s
    |
    C
    |\
    | \
    |  O
    |   \
    |    \
    A-----B--> r

Vertices in unit coordinates::

    O = ( 0,  0)
    A = (-1, -1)
    B = ( 1, -1)
    C = (-1,  1)

Equilateral coordinates :math:`(x, y)`::

          C
         / \
        /   \
       /     \
      /   O   \
     /         \
    A-----------B

Vertices in equilateral coordinates::

    O = ( 0,          0)
    A = (-1, -1/sqrt(3))
    B = ( 1, -1/sqrt(3))
    C = ( 0,  2/sqrt(3))

.. _tet-coords:

Coordinates on the tetrahedron
------------------------------

Unit coordinates :math:`(r, s, t)`::

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

Vertices in unit coordinates :math:`(r, s, t)`::

    O = ( 0,  0,  0)
    A = (-1, -1, -1)
    B = ( 1, -1, -1)
    C = (-1,  1, -1)
    D = (-1, -1,  1)

Vertices in equilateral coordinates :math:`(x, y, z)`::

    O = ( 0,          0,          0)
    A = (-1, -1/sqrt(3), -1/sqrt(6))
    B = ( 1, -1/sqrt(3), -1/sqrt(6))
    C = ( 0,  2/sqrt(3), -1/sqrt(6))
    D = ( 0,          0,  3/sqrt(6))

Hypercubes
^^^^^^^^^^

.. autoclass:: Hypercube

.. _square-coords:

Coordinates on the square
-------------------------

Unit coordinates on :math:`(r, s)`::

     ^ s
     |
     C---------D
     |         |
     |         |
     |    O    |
     |         |
     |         |
     A---------B --> r


Vertices in unit coordinates::

    O = ( 0,  0)
    A = (-1, -1)
    B = ( 1, -1)
    C = (-1,  1)
    D = ( 1,  1)

.. _cube-coords:

Coordinates on the cube
-----------------------

Unit coordinates on :math:`(r, s, t)`::

    t
    ^
    |
    E----------G
    |\         |\
    | \        | \
    |  \       |  \
    |   F------+---H
    |   |  O   |   |
    A---+------C---|--> s
     \  |       \  |
      \ |        \ |
       \|         \|
        B----------D
         \
          v r

Verties in unit coordinates::

    O = ( 0,  0,  0)
    A = (-1, -1, -1)
    B = ( 1, -1, -1)
    C = (-1,  1, -1)
    D = ( 1,  1, -1)
    E = (-1, -1,  1)
    F = ( 1, -1,  1)
    G = (-1,  1,  1)
    H = ( 1,  1,  1)
