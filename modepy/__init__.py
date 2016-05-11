from __future__ import division

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


from modepy.modes import (
        jacobi, grad_jacobi, simplex_onb, grad_simplex_onb,
        simplex_monomial_basis, grad_simplex_monomial_basis,
        simplex_best_available_basis, grad_simplex_best_available_basis)
from modepy.nodes import equidistant_nodes, warp_and_blend_nodes
from modepy.matrices import (vandermonde,
        resampling_matrix, differentiation_matrices,
        inverse_mass_matrix, mass_matrix,
        modal_face_mass_matrix, nodal_face_mass_matrix)
from modepy.quadrature import Quadrature, QuadratureRuleUnavailable
from modepy.quadrature.jacobi_gauss import (
        JacobiGaussQuadrature, LegendreGaussQuadrature)
from modepy.quadrature.xiao_gimbutas import XiaoGimbutasSimplexQuadrature
from modepy.quadrature.vioreanu_rokhlin import VioreanuRokhlinSimplexQuadrature
from modepy.quadrature.grundmann_moeller import GrundmannMoellerSimplexQuadrature

from modepy.version import VERSION_TEXT as __version__

__all__ = [
        "__version__",

        "jacobi", "grad_jacobi",
        "simplex_onb", "grad_simplex_onb",
        "simplex_monomial_basis", "grad_simplex_monomial_basis",
        "simplex_best_available_basis", "grad_simplex_best_available_basis",

        "equidistant_nodes", "warp_and_blend_nodes",

        "vandermonde", "resampling_matrix", "differentiation_matrices",
        "inverse_mass_matrix", "mass_matrix", "modal_face_mass_matrix",
        "nodal_face_mass_matrix",

        "Quadrature", "QuadratureRuleUnavailable",
        "JacobiGaussQuadrature", "LegendreGaussQuadrature",
        "XiaoGimbutasSimplexQuadrature", "GrundmannMoellerSimplexQuadrature",
        "VioreanuRokhlinSimplexQuadrature",
        ]

from pytools import MovedFunctionDeprecationWrapper

get_simplex_onb = MovedFunctionDeprecationWrapper(simplex_onb)
get_grad_simplex_onb = MovedFunctionDeprecationWrapper(grad_simplex_onb)
get_warp_and_blend_nodes = MovedFunctionDeprecationWrapper(warp_and_blend_nodes)
