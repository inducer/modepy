from __future__ import annotations

from importlib import metadata
from urllib.request import urlopen


_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2013-24 Andreas Klöckner and contributors"

release = metadata.version("modepy")
version = ".".join(release.split(".")[:2])

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pymbolic": ("https://documen.tician.de/pymbolic", None),
    "python": ("https://docs.python.org/3/", None),
    "pytools": ("https://documen.tician.de/pytools", None),
}

autodoc_member_order = "bysource"
