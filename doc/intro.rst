Introduction
============

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

Points in unit coordinates::

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

Points in equilateral coordinates::

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

Points in unit coordinates::

    O=( 0, 0, 0)
    A=(-1,-1,-1)
    B=(+1,-1,-1)
    C=(-1,+1,-1)
    D=(-1,-1,+1)

Points in equilateral coordinates :math:`(x,y,z)`::

    O = (0,0,0)
    A = (-1,-1/sqrt(3),-1/sqrt(6))
    B = ( 1,-1/sqrt(3),-1/sqrt(6))
    C = ( 0, 2/sqrt(3),-1/sqrt(6))
    D = ( 0,         0, 3/sqrt(6))

