"""
The associated paper can be found at

    R. S. Womersley,
    *Efficient Spherical Designs With Good Geometric Properties*,
    Springer International Publishing, pp. 1243--1285, 2018,
    `DOI <https://doi.org/10.1007/978-3-319-72456-0_57>`__.

The quadrature rules were extracted from
    https://web.maths.unsw.edu.au/~rsw/Sphere/EffSphDes
"""

from __future__ import annotations

import pathlib
import re
import sys
import tempfile
from typing import TypedDict

import numpy as np


# {{{ download rules

WOMERSLEY_SF_URL = "https://web.maths.unsw.edu.au/~rsw/Sphere/Points/SF/SF29-Nov-2012.zip"
WOMERSLEY_SS_URL = "https://web.maths.unsw.edu.au/~rsw/Sphere/Points/SS/SS31-Mar-2016.zip"

# NOTE: filenames have the formad `sfDDD.NNNNN` where `DDD` is the degree and
# `NNNNN` is the number of points in the quadrature rule
RE_DEGREE = re.compile(r"[sf]{2}(\d{3})\.(\d{5})")


def download_from_web_if_not_present(url: str, local_name: str | None = None) -> None:
    from os.path import basename, exists

    if local_name is None:
        local_name = basename(url)

    if exists(local_name):
        return

    import ssl

    # FIXME: downloading from WOMERSLEY_*_URL gives an SSL error sometimes
    context = ssl._create_unverified_context()

    from importlib import metadata

    try:
        version = metadata.version("modepy")
    except metadata.PackageNotFoundError:
        version = "0.0"

    from urllib.request import Request, urlopen

    request = Request(url, headers={"User-Agent": f"modepy/{version}"})
    with urlopen(request, timeout=120, context=context) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            size = int(content_length)

        downloaded = 0
        chunk_size = 8192

        with open(local_name, "wb") as outf:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break

                outf.write(chunk)
                downloaded += len(chunk)

                if content_length is not None:
                    sys.stdout.write(
                        f"\rDownloaded {downloaded // 1024}KB / {size // 1024}KB"
                    )
                else:
                    sys.stdout.write(f"\rDownloaded {downloaded / 1025}KB")
                sys.stdout.flush()

        print("\n")


class Rule(TypedDict):
    quad_degree: int
    points: np.ndarray[tuple[int, ...], np.dtype[np.float64]]


def extract_rules(url: str, *, cwd: pathlib.Path) -> dict[int, Rule]:
    filename = cwd / pathlib.Path(url).name

    print(f"INFO: Downloading archive from '{url}': '{filename}'.")
    download_from_web_if_not_present(url, local_name=filename)

    import zipfile

    with zipfile.ZipFile(filename, "r") as z:
        z.extractall(cwd)
    print(f"INFO: Extracted archive to '{cwd / filename.stem}'.")

    rules = {}
    dirname = cwd / filename.stem
    for name in dirname.iterdir():
        match = RE_DEGREE.match(name.name)
        if not match:
            print(f"ERROR: Invalid rule file name: '{name}'")
            continue

        degree, npoints = [int(g) for g in match.groups()]
        points = np.loadtxt(name).T
        assert points.shape == (3, npoints)

        rules[degree] = Rule(quad_degree=degree, points=points)

    print(f"INFO: Extracted {len(rules)} rules from '{url}'.")
    return rules


# }}}


# {{{ main


def extract_womersley_quadrature_rules(
    outfile: pathlib.Path,
    *,
    force: bool = False,
) -> int:
    if not force and outfile.exists():
        print(f"ERROR: File already exists (pass --force): '{outfile}'")
        return 1

    cwd = pathlib.Path(tempfile.gettempdir())
    sf_rules = extract_rules(WOMERSLEY_SF_URL, cwd=cwd)
    ss_rules = extract_rules(WOMERSLEY_SS_URL, cwd=cwd)

    np.savez(outfile, sf_rules=sf_rules, ss_rules=ss_rules)

    return 0


# }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", type=pathlib.Path)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    raise SystemExit(
        extract_womersley_quadrature_rules(
            outfile=args.outfile,
            force=args.force,
        )
    )
