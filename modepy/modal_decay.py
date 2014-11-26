from __future__ import division
from __future__ import absolute_import
from six.moves import range

__copyright__ = "Copyright (C) 2010-2012 Andreas Kloeckner"

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

import numpy as np
import numpy.linalg as la

__doc__ = """Estimate the smoothness of a function represented in a basis
returned by :func:`modepy.simplex_onb`.

The method implemented in this module follows this article:

    A. Kloeckner, T. Warburton, and J. S. Hesthaven. "Viscous Shock Capturing
    in a Time-Explicit Discontinuous Galerkin Method". Mathematical Modelling
    of Natural Phenomena 6, No. 3 (May 16, 2011): 57-83.
    http://dx.doi.org/10.1051/mmnp/20116303
    http://arxiv.org/abs/1102.3190

.. versionadded:: 2013.2
"""




def make_mode_number_vector(mode_order_tuples, ignored_modes):
    node_cnt = len(mode_order_tuples)

    mode_number_vector = np.zeros(node_cnt-ignored_modes, dtype=np.int)
    for i, mid in enumerate(mode_order_tuples):
        if i < ignored_modes:
            continue
        mode_number_vector[i-ignored_modes] = sum(mid)

    return mode_number_vector



def create_decay_baseline(mode_number_vector, n):
    """Create a vector of modal coefficients that exhibit 'optimal'
    (:math:`k^{-N}`) decay.
    """

    zeros = mode_number_vector == 0

    modal_coefficients = mode_number_vector**(-n)
    modal_coefficients[zeros] = 1 # irrelevant, just keeps log from NaNing

    modal_coefficients /= la.norm(modal_coefficients)

    return modal_coefficients




def get_decay_fit_matrix(mode_number_vector, ignored_modes, weight_vector):
    a = np.zeros((len(mode_number_vector), 2), dtype=np.float64)
    a[:,0] = weight_vector
    a[:,1] = weight_vector * np.log(mode_number_vector)

    if ignored_modes == 0:
        assert not np.isfinite(a[0,1])
        a[0,1] = 0

    return la.pinv(a)




def skyline_pessimize(modal_values):
    nelements, nmodes = modal_values.shape

    result = np.empty_like(modal_values)

    for iel in range(nelements):
        my_modes = modal_values[iel]
        cur_val = max(my_modes[-1], my_modes[-2])

        for imode in range(nmodes-1, -1, -1):
            if my_modes[imode] > cur_val:
                cur_val = my_modes[imode]

            result[iel, imode] = cur_val

    return result




def fit_modal_decay(coeffs, dims, n, ignored_modes=1):
    """Fit a curve to the modal decay on each element.

    :arg coeffs: a array of shape *(num_elements, num_modes)* containing modal
        coefficients of the functions to be analyzed
    :arg dims: number of dimensions
    :arg ignored_modes: the number of modal coefficients to ignore at the
        beginning. The default value of '1' ignores the constant.
    :returns: a tuple *(expn, constant)* of arrays of length *num_elements*,
        where the modal decay is fit to the curve 
        ``constant * total_degree**exponent``.

    ``-exponent-1`` can be used as a rough indicator of how many continuous
    derivatives the underlying function possesses.
    """

    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
            as gnitstam

    mode_order_tuples = list(gnitstam(n, dims))

    coeffs_squared = skyline_pessimize(coeffs**2)

    mode_number_vector_int = make_mode_number_vector(mode_order_tuples, ignored_modes)
    mode_number_vector = mode_number_vector_int.astype(np.float64)
    weight_vector = np.ones_like(mode_number_vector)

    fit_mat = get_decay_fit_matrix(mode_number_vector, ignored_modes,
            weight_vector)

    el_norm_squared = np.sum(coeffs_squared, axis=-1)
    scaled_baseline = (
            create_decay_baseline(mode_number_vector, n)
            * el_norm_squared[:, np.newaxis])**2
    log_modal_coeffs = np.log(coeffs_squared[:, ignored_modes:] + scaled_baseline)/2

    assert fit_mat.shape[0] == 2 # exponent and log(constant)

    fit_values = np.dot(fit_mat, (weight_vector*log_modal_coeffs).T).T

    exponent = fit_values[:, 1]
    const = np.exp(fit_values[:, 0])

    if 0:
        import matplotlib.pyplot as pt
        pt.plot(log_modal_coeffs.flat, "o-")

        fit = np.log(const[:, np.newaxis] * mode_number_vector**exponent[:, np.newaxis])

        pt.plot(fit.flat)

        #plot_expt = np.zeros_like(log_modal_coeffs)
        #plot_expt[:] = exponent[:, np.newaxis]
        #pt.plot(plot_expt.flat)

        pt.show()

    return exponent, const




def estimate_relative_expansion_residual(coeffs, dims, n, ignored_modes=1):
    """Use the modal fit to estimate the relative residual of the expansion.
    The arguments to this function exactly match :func:`fit_modal_decay`.

    :returns: An array of estimates of the fraction of the :math:`L^2` norm
        contained in the (unrepresented) tail of 

    An idea like this is described in this article:

        H. Feng and C. Mavriplis, "Adaptive Spectral Element Simulations of Thin
        Premixed Flame Sheet Deformations", Journal of Scientific Computing, Volume
        17, Nr. 1, S. 385-395, Dec. 2002.
        http://dx.doi.org/10.1023/A:1015137722700

    For this function, however, the decay curve is fitted using the
    Kloeckner/Warburton/Hesthaven technique (see above).
    """

    l2_norms = np.sqrt(np.sum(coeffs**2, axis=-1))

    exponent, const = fit_modal_decay(coeffs, dims, n, ignored_modes)

    # sqrt(integral from (n+1) to infty : (x**exponent)**2)
    residuals = const * np.sqrt((-(n+1)**(2*exponent+1)/(2*exponent+1)))

    result = residuals/l2_norms
    result[result > 1] = 0
    return result
