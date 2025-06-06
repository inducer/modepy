from __future__ import annotations


__copyright__ = "Copyright (C) 2009-2013 Andreas Kloeckner"

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


import logging

import numpy as np
import numpy.linalg as la
import pytest

from pymbolic.mapper.evaluator import EvaluationMapper
from pymbolic.mapper.stringifier import (
    CSESplittingStringifyMapperMixin,
    StringifyMapper,
)

import modepy as mp
from modepy.typing import ArrayF


logger = logging.getLogger(__name__)


@pytest.mark.parametrize(("alpha", "beta"), [
    (0, 0),              # Gauss-Legendre
    (-0.5, -0.5),        # Chebyshev-Gauss (first kind)
    (0.5, 0.5),          # Chebyshev-Gauss (second kind)
    (1, 0),
    (3, 2),
    (0, 2),
    (5, 0),
    (3, 4)
    ])
def test_scaled_jacobi(alpha, beta):
    from modepy.modes import binom, scaled_jacobi as jacobi

    for n in range(10):
        err1 = abs(jacobi(alpha, beta, n, 1) - binom(n + alpha, n))
        errm1 = abs(jacobi(alpha, beta, n, -1) - (-1)**n*binom(n + beta, n))

        assert err1 < 1e-11
        assert errm1 < 1e-11


# {{{ test_orthonormality_jacobi_1d

@pytest.mark.parametrize(("alpha", "beta", "ebound"), [
    (0, 0, 5e-14),              # Gauss-Legendre
    (-0.5, -0.5, 7e-15),        # Chebyshev-Gauss (first kind)
    (0.5, 0.5, 9e-15),          # Chebyshev-Gauss (second kind)
    (1, 0, 4e-14),
    (3, 2, 3e-14),
    (0, 2, 3e-13),
    (5, 0, 3e-13),
    (3, 4, 1e-14)
    ])
def test_orthonormality_jacobi_1d(alpha, beta, ebound):
    """Verify that the Jacobi polymials are orthogonal in 1D."""
    from modepy.quadrature.jacobi_gauss import JacobiGaussQuadrature

    max_n = 10
    quad = JacobiGaussQuadrature(alpha, beta, 4*max_n, force_dim_axis=True)

    from functools import partial
    jac_f = [partial(mp.jacobi, alpha, beta, n) for n in range(max_n)]
    maxerr = 0

    for i, fi in enumerate(jac_f):
        for j, fj in enumerate(jac_f):
            result = quad(lambda x: fi(x)*fj(x))  # noqa: B023
            true_result = 1.0 if i == j else 0.0

            err = abs(result-true_result)
            maxerr = np.maximum(maxerr, err)
            if abs(result - true_result) > ebound:
                logger.error("[FAILED] (%g, %g): (%d, %d) error %.5e",
                        alpha, beta, i, j, abs(result - true_result))

            assert abs(result-true_result) < ebound

# }}}


@pytest.mark.parametrize("shape", [
    mp.Simplex(2),
    mp.Simplex(3),
    mp.Hypercube(2),
    mp.Hypercube(3),
    ])
@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_basis_size_against_space_dim(shape, order):
    space = mp.space_for_shape(shape, order)
    for basis in [
                mp.basis_for_space(space, shape),
                mp.orthonormal_basis_for_space(space, shape),
                mp.monomial_basis_for_space(space, shape),
            ]:
        assert len(basis.functions) == space.space_dim


# {{{ test_basis_orthogonality

@pytest.mark.parametrize(("order", "ebound"), [
    (1, 2e-15),
    (2, 7e-15),
    (3, 1e-14),
    # (4, 3e-14),
    # (7, 3e-14),
    # (9, 2e-13),
    ])
@pytest.mark.parametrize("shape", [
    mp.Simplex(2),
    mp.Simplex(3),
    mp.Hypercube(2),
    mp.Hypercube(3),
    ])
def test_basis_orthogonality(shape, order, ebound):
    """Test orthogonality of ONBs using cubature."""

    qspace = mp.space_for_shape(shape, 2*order)
    cub = mp.quadrature_for_space(qspace, shape)
    basis = mp.orthonormal_basis_for_space(mp.space_for_shape(shape, order), shape)

    maxerr = 0
    for i, f in enumerate(basis.functions):
        for j, g in enumerate(basis.functions):
            true_result = 1 if i == j else 0
            result = cub(lambda x: f(x)*g(x))  # noqa: B023
            err = abs(result-true_result)
            logger.info("error %.5e max %.5e", err, maxerr)
            maxerr = np.maximum(maxerr, err)
            if err > ebound:
                logger.info("bound exceeded at order %d for (f_{%d}, f_{%d}): %.5e",
                        order, i, j, err)
            assert err < ebound

    logger.info("order %d max error %.5e", order, maxerr)

# }}}


# {{{ test_basis_grad

def get_inhomogeneous_tensor_prod_basis(space, shape):
    if space.spatial_dim == 1:
        return mp.basis_for_space(space, shape)

    orders = (space.order, 2, 7)[:space.spatial_dim]
    space = mp.space_for_shape(shape, orders)
    return mp.basis_for_space(space, shape)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [5, 8])
@pytest.mark.parametrize(("shape_cls", "basis_getter"), [
    (mp.Simplex, mp.basis_for_space),
    (mp.Simplex, mp.orthonormal_basis_for_space),
    (mp.Simplex, mp.monomial_basis_for_space),

    (mp.Hypercube, mp.basis_for_space),
    (mp.Hypercube, mp.orthonormal_basis_for_space),
    (mp.Hypercube, mp.monomial_basis_for_space),

    (mp.Hypercube, get_inhomogeneous_tensor_prod_basis),
    ])
def test_basis_grad(dim, shape_cls, order, basis_getter):
    """Do a simplistic FD-style check on the gradients of the basis."""

    h = 1.0e-4

    shape = shape_cls(dim)
    rng = np.random.Generator(np.random.PCG64(17))
    basis = basis_getter(mp.space_for_shape(shape, order), shape)

    from pytools import wandering_element
    from pytools.convergence import EOCRecorder
    for ifunc, (bf, gradbf) in enumerate(
                zip(basis.functions, basis.gradients, strict=True)):
        eoc_rec = EOCRecorder()
        for h in [1e-2, 1e-3]:
            r = mp.random_nodes_for_shape(shape, nnodes=1000, rng=rng)

            gradbf_v = np.array(gradbf(r))
            gradbf_v_num = np.array([
                (bf(r+h*unit) - bf(r-h*unit))/(2*h)
                for unit_tuple in wandering_element(shape.dim)
                for unit in (np.array(unit_tuple).reshape(-1, 1),)
                ])

            ref_norm = la.norm((gradbf_v).reshape(-1), np.inf)
            err = la.norm((gradbf_v_num - gradbf_v).reshape(-1), np.inf)
            if ref_norm > 1e-13:
                err = err/ref_norm

            logger.info("error: %.5e", err)
            eoc_rec.add_data_point(h, err)

            for idiff_axis in range(shape.dim):
                dr_bf = basis.derivatives(idiff_axis)[ifunc](r)
                ref = gradbf_v[idiff_axis]
                err = la.norm(dr_bf - ref, 2)
                if la.norm(ref, 2) > 1e-13:
                    err = err/la.norm(ref, 2)

                assert err < 1e-13

        tol = 1e-8
        if eoc_rec.max_error() >= tol:
            logger.info("\n%s", str(eoc_rec))

        assert (eoc_rec.max_error() < tol or eoc_rec.order_estimate() >= 1.5)

# }}}


# {{{ test symbolic modes

class MyStringifyMapper(CSESplittingStringifyMapperMixin[[]], StringifyMapper[[]]):
    pass


class MyEvaluationMapper(EvaluationMapper[ArrayF]):
    def map_if(self, expr):
        return np.where(self.rec(expr.condition),
                self.rec(expr.then), self.rec(expr.else_))


@pytest.mark.parametrize("shape", [
    mp.Simplex(1),
    mp.Simplex(2),
    mp.Simplex(3),
    mp.Hypercube(1),
    mp.Hypercube(2),
    mp.Hypercube(3),
    ])
@pytest.mark.parametrize("order", [5, 8])
@pytest.mark.parametrize("basis_getter", [
    (mp.basis_for_space),
    (mp.orthonormal_basis_for_space),
    (mp.monomial_basis_for_space),
    ])
def test_symbolic_basis(shape, order, basis_getter):
    basis = basis_getter(mp.space_for_shape(shape, order), shape)
    sym_basis = [mp.symbolicize_function(f, shape.dim) for f in basis.functions]

    # {{{ test symbolic against direct eval

    logger.info(75*"#")
    logger.info("VALUES")
    logger.info(75*"#")

    rng = np.random.Generator(np.random.PCG64(17))
    r = mp.random_nodes_for_shape(shape, 10000, rng=rng)

    for func, sym_func in zip(basis.functions, sym_basis, strict=True):
        strmap = MyStringifyMapper()
        s = strmap(sym_func)
        for name, val in strmap.cse_name_list:
            logger.info("%s <- %s", name, val)
        logger.info("sym_func: %s", s)

        sym_val = MyEvaluationMapper({"r": r, "abs": abs})(sym_func)
        ref_val = func(r)

        ref_norm = la.norm(ref_val, np.inf)
        err = la.norm(sym_val-ref_val, np.inf)
        if ref_norm:
            err = err/ref_norm

        logger.info("ERROR: %.5e", err)
        logger.info("\n")

        assert np.allclose(sym_val, ref_val)

    # }}}

    # {{{ test gradients

    logger.info(75*"#")
    logger.info("GRADIENTS")
    logger.info(75*"#")

    sym_grad_basis = [mp.symbolicize_function(f, shape.dim) for f in basis.gradients]

    for grad, sym_grad in zip(basis.gradients, sym_grad_basis, strict=True):
        strmap = MyStringifyMapper()
        s = strmap(sym_grad)
        for name, val in strmap.cse_name_list:
            logger.info("%s <- %s", name, val)
        logger.info("sym_grad: %s", s)

        sym_val = MyEvaluationMapper({"r": r, "abs": abs})(sym_grad)
        ref_val = grad(r)
        if not isinstance(ref_val, tuple):
            assert not isinstance(sym_val, tuple)
            sym_val = (sym_val,)
            ref_val = (ref_val,)

        for sv_i, rv_i in zip(sym_val, ref_val, strict=True):
            ref_norm = la.norm(rv_i, np.inf)
            err = la.norm(sv_i-rv_i, np.inf)
            if ref_norm:
                err = err/ref_norm

            logger.info("ERROR: %.5e", err)
            logger.info("\n")

            assert np.allclose(sv_i, rv_i)

    # }}}

# }}}


# {{{ test_modal_coeffs_by_projection

@pytest.mark.parametrize("dim", [2, 3])
def test_modal_coeffs_by_projection(dim):
    shape = mp.Simplex(dim)
    space = mp.space_for_shape(shape, order=5)
    basis = mp.orthonormal_basis_for_space(space, shape)

    quad = mp.XiaoGimbutasSimplexQuadrature(10, dim)
    assert quad.exact_to >= 2*space.order

    rng = np.random.default_rng(seed=42)
    modal_coeffs = rng.normal(size=space.space_dim)
    vdm = mp.vandermonde(basis.functions, quad.nodes)

    evaluated = vdm @ modal_coeffs

    modal_coeffs_2 = vdm.T @ (evaluated*quad.weights)

    diff = modal_coeffs - modal_coeffs_2

    assert la.norm(diff, 2) < 3e-13

# }}}


def test_tp_0d():
    basis = mp.TensorProductBasis([])
    nodes = np.zeros((0, 15))
    for f in basis.functions:
        res = f(nodes)
        assert res.shape == (15,)

    for f in basis.gradients:
        res = f(nodes)
        assert len(res) == 0


# You can test individual routines by typing
# $ python test_modes.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
