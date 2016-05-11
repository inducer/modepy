Installation
============

This command should install :mod:`modepy`::

    pip install modepy

You may need to run this with :command:`sudo`.
If you don't already have `pip <https://pypi.python.org/pypi/pip>`_,
run this beforehand::

    curl -O https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    python get-pip.py

For a more manual installation, download the source, unpack it,
and say::

    python setup.py install

In addition, you need to have :mod:`numpy` installed.

User-visible Changes
====================

Version 2016.1
--------------

.. note::

    This version is currently under development. You can get snapshots from
    ModePy's `git repository <https://github.com/inducer/modepy>`_

* Add monomial modes in :mod:`modepy.modes`.

Version 2013.3
--------------

* Add :class:`modepy.VioreanuRokhlinSimplexQuadrature`.
* Update nodes and weights for :class:`modepy.XiaoGimbutasSimplexQuadrature`.
  (thanks to Zydrunas Gimbutas)

Version 2013.2.1
----------------

* Minor Py3 test fix.

Version 2013.2
--------------

* Add :func:`modepy.tools.estimate_lebesgue_constant` to public interface.
* Add :mod:`modepy.modal_decay`.
* Add :func:`modepy.resampling_matrix`.
* Add :func:`modepy.equidistant_nodes`.
* Add :func:`modepy.differentiation_matrices`.
* Remove ``get_`` prefix from a number of functions. (Backward compatible wrappers provided.)

Version 2013.1.1
----------------

* Allow passing *node_tuples* to :func:`modepy.warp_and_blend_nodes` in 1D.

Version 2013.1
--------------

* Initial release.

.. _license:

License
=======

:mod:`modepy` is licensed to you under the MIT/X Consortium license:

Copyright (c) 2012-13 Andreas Klöckner, Tim Warburton, Jan Hesthaven, Xueyu Zhu

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

Frequently Asked Questions
==========================

The FAQ is maintained collaboratively on the
`Wiki FAQ page <http://wiki.tiker.net/ModePy/FrequentlyAskedQuestions>`_.

