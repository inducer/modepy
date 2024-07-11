r"""
The quadrature rules extracted here can be found in the supplemental materials
from

    F. D. Witherden, P. E. Vincent,
    On the Identification of Symmetric Quadrature Rules for Finite Element Methods,
    Computers & Mathematics with Applications, Vol. 69, pp. 1232--1241, 2015,
    https://doi.org/10.1016/j.camwa.2015.03.017

They are available under a Creative Commons license.
"""

import os

import numpy as np

from pytools import download_from_web_if_not_present


_PYTHON_TEMPLATE = """# GENERATED by modepy/contrib/weights-from-witherden-vincent.py
# DO NOT EDIT

__copyright__ = '(C) 2015 F. D. Witherden, P. E. Vincent'

__license__ = \'\'\'
This work is licensed under a Creative Commons Attribution
4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
\'\'\'

import numpy as np


def process_rules(rules):
    result = {}
    for order, rule in rules.items():
        result[order] = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in rule.items()
            }

    return result


quad_data = process_rules(%s)

hex_data = process_rules(%s)"""


def generate_witherden_vincent_quadrature_rules(outfile):
    filename = "witherden_vincent.zip"
    download_from_web_if_not_present(
            url="https://ars.els-cdn.com/content/image/1-s2.0-S0898122115001224-mmc1.zip",
            local_name=filename)

    import zipfile
    with zipfile.ZipFile(filename, "r") as zp:
        zp.extractall(filename[:-4])

    # files are named <strength>-<number-of-nodes>.txt
    # contents of each row are <x> <y> [<z>] <weight>

    quad_rules = {}
    dirname = os.path.join(filename[:-4], "expanded", "quad")
    for f in os.listdir(dirname):
        degree = int(f.split("-")[0])
        txt = np.loadtxt(os.path.join(dirname, f)).reshape(-1, 3)

        quad_rules[degree] = {
                "quad_degree": degree,
                "points": txt[:, :-1].T.tolist(),
                "weights": txt[:, -1].tolist()
                }

        assert abs(np.sum(txt[:, -1]) - 4.0) < 1.0e-12

    hex_rules = {}
    dirname = os.path.join(filename[:-4], "expanded", "hex")
    for f in os.listdir(dirname):
        degree = int(f.split("-")[0])
        txt = np.loadtxt(os.path.join(dirname, f)).reshape(-1, 4)

        hex_rules[degree] = {
                "quad_degree": degree,
                "points": txt[:, :-1].T.tolist(),
                "weights": txt[:, -1].tolist()
                }

        assert abs(np.sum(txt[:, -1]) - 8.0) < 1.0e-12

    from pprint import pformat
    txt = (_PYTHON_TEMPLATE % (
        pformat(quad_rules), pformat(hex_rules)
        )).replace('"', "")

    if outfile:
        with open(outfile, "w") as fd:
            fd.write(txt)
    else:
        print(txt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", nargs="?", default="")
    args = parser.parse_args()

    generate_witherden_vincent_quadrature_rules(outfile=args.outfile)
