"""
.. autoclass:: ArrayF
.. autoclass:: RealValue
.. data:: RealValueT

    A type variable with an upper bound similar to :class:`RealValue`.


.. autoclass:: BasisFunction
.. autoclass:: BasisGradient
.. autoclass:: NodalFunction
"""

from __future__ import annotations


__copyright__ = "Copyright (C) 2025 University of Illinois Board of Trustees"


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

from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
    cast,
)

import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:
    import pymbolic.primitives


# NOTE: We're currently using NDArray[np.floating] throughout, even
# though most code should be just fine with complex values.
# https://github.com/microsoft/pyright/discussions/10474
ArrayF: TypeAlias = NDArray[np.floating]

RealValue: TypeAlias = """
        ArrayF
        | pymbolic.primitives.ExpressionNode
        | float
        | np.floating
        """


RealValueT = TypeVar("RealValueT",
                     ArrayF,
                     "pymbolic.primitives.ExpressionNode",
                     float,
                     np.floating,
                 )


BasisFunction: TypeAlias = Callable[[ArrayF], ArrayF]
BasisGradient: TypeAlias = Callable[[ArrayF], tuple[ArrayF, ...]]


NodalFunction: TypeAlias = Callable[[ArrayF], ArrayF]


NumpyTypeT = TypeVar("NumpyTypeT", bound=np.generic)


def is_array(x: NDArray[NumpyTypeT] | object) -> NDArray[NumpyTypeT]:
    assert isinstance(x, np.ndarray)
    return cast("NDArray[NumpyTypeT]", x)
