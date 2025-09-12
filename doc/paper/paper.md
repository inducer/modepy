---
breaks: false
title: 'modepy: Basis Functions, Interpolation, and Quadrature (not just) for Finite Elements'
tags:
  - Python
  - approximation theory
authors:
  - given-names: Andreas
    surname: Klöckner
    corresponding: true
    orcid: 0000-0003-1228-519X
    affiliation: 1
  - given-names: Alexandru
    surname: Fikl
    orcid: 0000-0002-0552-5936
    affiliation: 2
  - given-names: Jacob Xiaoyu
    surname: Wei
    orcid: 0000-0001-7063-7865
    affiliation: 3
  - given-names: Thomas H.
    surname: Gibson
    orcid: 0000-0002-7978-6848
    affiliation: 4
  - given-names: Addison J.
    surname: Alvey-Blanco
    affiliation: 1
affiliations:
  - index: 1
    name: Siebel School of Computing and Data Science, University of Illinois at Urbana-Champaign, US
  - index: 2
    name: Institute for Advanced Environmental Research (ICAM), West University of Timişoara, Romania
  - index: 3
    name: Pathlit, US
  - index: 4
    name: Advanced Micro Devices Inc., US
date: 1 May 2025
bibliography: paper.bib

---

# Summary

`modepy` provides a means for constructing geometric shapes for reference
elements, equipping them with appropriate approximation spaces, and numerically
performing calculus operations (derivatives, integrals) on those spaces.

`modepy` focuses on high-order accuracy -- given an element size $h$, this
refers to the asymptotic decay of the approximation error as $O(h^n)$, for $n
\ge 3$, assuming sufficient smoothness of the solution being approximated. If
we consider a problem in $d$ dimensions, the number of unknowns scales as
$O(h^{-d})$. In the absence of high-order approximation, say at $n=1$, even
modest accuracy increases (e.g. an additional significant digit) can incur
significant cost (a $1000\times$ increase in the problem size in 3D).
Therefore, if accuracy is desired at manageable cost, high-order methods are
crucial.

A popular approach for accurate approximation of functions on geometrically
complex domains is the use of *unstructured discretizations*, which represent
the geometry as a typically disjoint union (a "mesh") of primitive geometric
shapes, most often simplices and quadrilaterals. Given the means to perform
calculus operations on these *reference elements* and mapping functions from
them to the *global* elements, calculus operations become available on the
entire domain. Notably, one can approximate the reference-to-global coordinate
mapping function itself with the same high-order machinery. Altogether, this
paves the way for the solution of integral and (partial) differential
equations. Those, in turn, can be used to model many phenomena in the physical
world, including fluid flow, electromagnetism, and solid mechanics. The finite
element method (FEM) is a popular example of this, including its continuous and
discontinuous flavors. Further examples include collocation, spectral, and
Nyström methods.

`modepy` has been used to construct FEM solvers [@Grudge; @PyNucleus] and
integral equation solvers [@Pytential]. It is written in pure modern Python 3,
offering comprehensive type annotations and full documentation with minimal
runtime dependencies, of which numpy is perhaps the most significant. Versions
going back to as early as 2013 offer broad compatibility with older versions of
Python, including Python 2.7.

`modepy` is licensed under the MIT license and available on Github at
<https://github.com/inducer/modepy/>.

# Statement of need

High-order accurate calculus operations on unstructured discretizations are
crucial to many solvers of differential and integral equations, but their uses
can also extend to computer graphics and Computer-Aided Design (CAD). This
functionality is often embedded in an ad-hoc manner in larger codes, leading to
limited scope and a lack of reusability. `modepy` addresses this need by
providing a robust, reusable, generalizable, and composable implementation.
FIAT [@FIAT] is an early software tool aiming to address a similar need. It
offers a comprehensive set of finite elements and basis functions, but largely
focuses on tabulation. FInAT [@FInAT], a somewhat recent refinement, adds a
focus on algebraic expressions for basis functions, exposing details needed for
efficient tensor product evaluation. Both are, however, tightly interwoven into
the FENiCS/Firedrake ecosystems (e.g. by depending on UFL, the "Unified Form
Language" used for the expression of variational forms). `StartUpDG.jl`
[@StartUpDG] provides functionality with some overlap with `modepy`, but with a
focus on the needs of discontinuous Galerkin FEM. `QuadPy` [@QuadPy] also has
some overlap with `modepy`, in that it can provide quadrature rules, and, while
it has comprehensive coverage, it is regrettably no longer open-source, and it
lacks `modepy`'s composability. `minterpy` [@Wicaksono2025], meanwhile, deals
exclusively with polynomial interpolation, with a focus on sparse grids.

A significant complicating factor in providing a flexible implementation is
that numerical solvers for differential equations (and likely many other
candidate users) must satisfy rigorous cost constraints, as simulation fidelity
is always traded off against computational cost. As a result, such software
commonly adopts cutting-edge high-performance computing (HPC) techniques, such
as GPU computation and distributed-memory approaches. To facilitate separation
of those concerns from the numerical methods, `modepy` adopts a two-pronged
approach. In many cases, it suffices for operations to be represented as data
in matrix or tabular form, so that no actual execution of `modepy` code is
needed in a cost-constrained setting.

A prominent exception to this is the evaluation of basis functions. To
facilitate this, `modepy` allows its function evaluation to be "traced" (in the
sense of lazy/deferred evaluation), so that an expression graph can be
obtained. This graph is represented with the help of the `pymbolic` [@Pymbolic]
software library. This, in turn, can interoperate with Python ASTs
[@Python_AST], SymPy [@SymPy], SymEngine [@SymEngine], as well as a number of
other pieces of software for symbolic computation. Another case where a purely
data-driven approach falls short is the setting of tensor-product elements. In
this case, for example, a mass matrix $\boldsymbol M$ with entries
$$
M_{ij}=\int_{[-1,1]^d} \phi_i(\boldsymbol r)\phi_j(\boldsymbol r) \mathrm{d}\boldsymbol r
$$
permits a Kronecker product factorization
$$
\boldsymbol M = \boldsymbol M^{1D}\otimes \cdots \otimes \boldsymbol M^{1D},
\qquad \text{with} \qquad
M^{1D}_{ij} = \int_{[-1,1]} \phi_i(r)\phi_j(r) \mathrm{d}r,
$$
assuming a suitable index order of nodes $\boldsymbol{r}_i$ and basis functions
$\{\phi_i\}$. In $d$ dimensions, the use of this factorization reduces the
asymptotic complexity of a matrix-vector product $\boldsymbol M \boldsymbol u$
from $O(N_p^{2d})$ to $O(N_p^{d+1})$, where $N_p$ is the number of degrees of
freedom in one dimension, and $d$ is the number of dimensions. To take
advantage of this factorization, an abstraction such as `modepy`'s must expose
additional information, including numbering of degrees of freedom. In
`modepy`'s case, this is accomplished via reshaping operation that converts
arrays of degrees of freedom back and forth between a flat (one-dimensional)
and a structured ($d$-dimensional) representation. This reshaping is applicable
to any array type (including GPU or other high-performance arrays) as long as
`numpy`-compatible reshaping is supported.

# Overview

The high-level concepts available in `modepy` are shapes (i.e. a reference
domain), modes (i.e. the basis functions), and nodes (i.e. the degrees of
freedom). These are implemented in a user-extensible fashion using the
`singledispatch` mechanism, with inspiration taken from common idiomatic usage
in Julia [@Bezanson2017]. 

## Shapes

The geometry of a reference element is described in `modepy` by the `Shape`
class. Built-in support exists for `Simplex` and `Hypercube` geometries,
encompassing the commonly used interval, triangle, tetrahedron, quadrilateral,
and hexahedral shapes (see \autoref{FigureSimplices}). `TensorProductShape` can
be used to compose additional shapes, that are useful in specific applications
and sometimes generated by meshing software (such as `gmsh` [@Geuzaine2009]). 

![Domains corresponding to the one-, two-, and three-dimensional simplices.](images/simplices.png){#FigureSimplices width="80%"}

## Modes and Spaces

To perform calculus operations, each reference element can be equipped with a
function space described by the `FunctionSpace` class. These represent a
finite-dimensional space of functions $\phi_i: D \to \mathbb{R}$, where $D$ is
the reference element domain, and no specific choice of basis.  Predefined
choices include the `PN` space, containing polynomials of total degree at most
`order`, and the `QN` space, containing polynomials of maximum degree at most
`order`. As with shapes, these spaces can be combined using
`TensorProductSpace`.

The basis is provided in the form of a `Basis` class, which allows access to
the basis functions, their derivatives (per axis) and opaque `mode_ids`. As
noted before, the basis functions and their derivatives can be evaluated
directly or can be traced using the `pymbolic` library to aid in code
generation. Various standard basis functions are provided, such as the
monomials, general Jacobi polynomials, and the
Proriol-Koornwinder-Dubiner-Owens (PKDO) basis from [@Dubiner1991] (see
\autoref{FigurePKDO}).

![PKDO basis functions for the triangle.](images/pkdo-2d.png){#FigurePKDO width="80%"}

## Nodes

A final component in an FEM discretization (or 'Ciarlet triple' [@Brenner2007,
Section 3.1]) is a set of 'degrees of freedom' ('DOFs') that uniquely define a
certain function in the span of a basis. `modepy` provides easy access to modal
DOFs (i.e. basis coefficients) and nodal DOFs (i.e. function or derivative
values at a point). As equispaced nodal DOFs are not usable in the high-order
setting [@Trefethen2020], `modepy` offers access to many types of nodes with
better properties for each reference element. It can also construct custom
nodes sets for a given set of basis functions in 2D via a procedure based on
eigenvalues of multiplication operators [@Vioreanu2014].

On simplices, the "warp-and-blend" nodes [@Warburton2007] are available, and on
the hypercube, standard tensor product nodes are constructed from
one-dimensional Legendre-Gauss(-Lobatto) nodes. `modepy` can also directly
interoperate with the `recursivenodes` library described in [@Isaac2020], which
offers additional well-conditioned nodes on the simplex.

## Quadrature

Besides the standard building blocks above, `modepy` also offers a wide array
of quadrature rules that can be used on each reference element. The quadrature
rules are provided as implementations of the `Quadrature` superclass. For the
interval, a broad selection of quadrature rules are present: Clenshaw--Curtis,
Fejér, and Jacobi-Gauss(-Lobatto). Additionally, several state-of-the-art
higher-dimensional rules are available:

* Grundmann--Möller [@Grundmann1978] rules for the $n$-simplex.
* Vioreanu--Rokhlin [@Vioreanu2014] rules for the two- and
  three-dimensional simplex (see \autoref{FigureQuadrature}). 
* Xiao--Gimbutas [@Xiao2010] rules for the two- and three-dimensional
  simplex.
* Jáskowiec--Sukumar [@Jaskowiec2021] rules for the tetrahedron.
* Witherden--Vincent [@Witherden2015] rules for the two- and
  three-dimensional hypercube (see \autoref{FigureQuadrature}).

![(left) Vioreanu--Rokhlin quadrature points of order 11 and (right) Witherden--Vincent quadrature points of order 11.](images/quadrature_rules.png){#FigureQuadrature width="50%"}

Most of the rules for simplices and hypercubes are available to high order
(i.e. $\ge 20$). However, `modepy` also has functionality that allows
constructing desired quadrature nodes on a given domain. This can be found in
`modepy.quadrature.construction`.

## Matrices

While `modepy` is sufficiently flexible to accommodate different applications,
it also provides functionality specific to FEM needs. In particular, it allows
constructing a variety of matrix operators that define necessary resampling
operators and bilinear forms used in FEM codes. A specific bilinear form matrix
can be obtained using `nodal_quadrature_bilinear_form_matrix`, which requires
providing the usual test functions, trial functions and corresponding nodes for
the input and output. This allows constructing very general families of
bilinear forms with or without oversampling. 

# Acknowledgements

A. Fikl was supported by the Office of Naval Research (ONR) as part of the
Multidisciplinary University Research Initiatives (MURI) Program, under Grant
Number *N00014-16-1-2617*. A. Klöckner was supported by the US National Science
Foundation under award number DMS-2410943, and by the US Department of Energy
under award number DE-NA0003963.

# References
