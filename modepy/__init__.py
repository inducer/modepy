__copyright__ = """Copyright (C) 2009, 2010, 2013 Andreas Kloeckner, Tim Warburton,
    Jan Hesthaven, Xueyu Zhu"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


from modepy.matrices import (
    diff_matrix_permutation, differentiation_matrices, inverse_mass_matrix,
    mass_matrix, modal_mass_matrix_for_face, nodal_mass_matrix_for_face,
    nodal_quad_mass_matrix_for_face, resampling_matrix, vandermonde)
from modepy.modes import (
    Basis, BasisNotOrthonormal, TensorProductBasis, basis_for_space, grad_jacobi,
    grad_legendre_tensor_product_basis, grad_simplex_best_available_basis,
    grad_simplex_monomial_basis, grad_simplex_onb, grad_tensor_product_basis, jacobi,
    legendre_tensor_product_basis, monomial_basis_for_space,
    orthonormal_basis_for_space, simplex_best_available_basis,
    simplex_monomial_basis, simplex_monomial_basis_with_mode_ids, simplex_onb,
    simplex_onb_with_mode_ids, symbolicize_function, tensor_product_basis)
from modepy.nodes import (
    edge_clustered_nodes_for_space, equidistant_nodes, equispaced_nodes_for_space,
    legendre_gauss_lobatto_tensor_product_nodes, node_tuples_for_space,
    random_nodes_for_shape, tensor_product_nodes, warp_and_blend_nodes)
from modepy.quadrature import (
    LegendreGaussTensorProductQuadrature, Quadrature, QuadratureRuleUnavailable,
    TensorProductQuadrature, ZeroDimensionalQuadrature, quadrature_for_space)
from modepy.quadrature.clenshaw_curtis import (
    ClenshawCurtisQuadrature, FejerQuadrature)
from modepy.quadrature.grundmann_moeller import GrundmannMoellerSimplexQuadrature
from modepy.quadrature.jacobi_gauss import (
    ChebyshevGaussQuadrature, GaussGegenbauerQuadrature, JacobiGaussQuadrature,
    LegendreGaussQuadrature)
from modepy.quadrature.jaskowiec_sukumar import JaskowiecSukumarQuadrature
from modepy.quadrature.vioreanu_rokhlin import VioreanuRokhlinSimplexQuadrature
from modepy.quadrature.witherden_vincent import WitherdenVincentQuadrature
from modepy.quadrature.xiao_gimbutas import XiaoGimbutasSimplexQuadrature
from modepy.shapes import (
    Face, Hypercube, Shape, Simplex, TensorProductShape, face_normal,
    faces_for_shape, submesh_for_shape, unit_vertices_for_shape)
from modepy.spaces import PN, QN, FunctionSpace, TensorProductSpace, space_for_shape
from modepy.version import VERSION_TEXT as __version__  # noqa: N811


GaussLegendreQuadrature = LegendreGaussQuadrature

__all__ = [
        "__version__",

        "Shape", "Face", "Simplex", "Hypercube", "TensorProductShape",
        "face_normal", "unit_vertices_for_shape", "faces_for_shape",
        "submesh_for_shape",

        "FunctionSpace", "TensorProductSpace", "PN", "QN", "space_for_shape",

        "jacobi", "grad_jacobi",
        "simplex_onb", "grad_simplex_onb", "simplex_onb_with_mode_ids",
        "simplex_monomial_basis", "grad_simplex_monomial_basis",
        "simplex_monomial_basis_with_mode_ids",
        "simplex_best_available_basis", "grad_simplex_best_available_basis",
        "tensor_product_basis", "grad_tensor_product_basis",
        "legendre_tensor_product_basis", "grad_legendre_tensor_product_basis",
        "symbolicize_function",

        "Basis", "BasisNotOrthonormal", "TensorProductBasis",
        "basis_for_space", "orthonormal_basis_for_space", "monomial_basis_for_space",

        "equidistant_nodes", "warp_and_blend_nodes",
        "tensor_product_nodes", "legendre_gauss_lobatto_tensor_product_nodes",
        "node_tuples_for_space",
        "edge_clustered_nodes_for_space", "equispaced_nodes_for_space",
        "random_nodes_for_shape",

        "vandermonde", "resampling_matrix", "differentiation_matrices",
        "diff_matrix_permutation",
        "inverse_mass_matrix", "mass_matrix",
        "modal_mass_matrix_for_face", "nodal_mass_matrix_for_face",
        "nodal_quad_mass_matrix_for_face",

        "Quadrature", "QuadratureRuleUnavailable",
        "ZeroDimensionalQuadrature",
        "TensorProductQuadrature", "LegendreGaussTensorProductQuadrature",
        "quadrature_for_space",

        "JacobiGaussQuadrature", "LegendreGaussQuadrature",
        "GaussLegendreQuadrature", "ChebyshevGaussQuadrature",
        "GaussGegenbauerQuadrature",
        "XiaoGimbutasSimplexQuadrature", "GrundmannMoellerSimplexQuadrature",
        "VioreanuRokhlinSimplexQuadrature",
        "ClenshawCurtisQuadrature", "FejerQuadrature",
        "JaskowiecSukumarQuadrature",

        "WitherdenVincentQuadrature",
        ]

from pytools import MovedFunctionDeprecationWrapper


get_simplex_onb = MovedFunctionDeprecationWrapper(simplex_onb)
get_grad_simplex_onb = MovedFunctionDeprecationWrapper(grad_simplex_onb)
get_warp_and_blend_nodes = MovedFunctionDeprecationWrapper(warp_and_blend_nodes)
