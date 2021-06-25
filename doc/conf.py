from urllib.request import urlopen

_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2013-21 Andreas Klöckner and contributors"

ver_dic = {}
exec(compile(open("../modepy/version.py").read(), "../modepy/version.py",
    "exec"), ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])
# The full version, including alpha/beta/rc tags.
release = ver_dic["VERSION_TEXT"]

intersphinx_mapping = {
        "https://docs.python.org/3/": None,
        "https://numpy.org/doc/stable/": None,
        "https://documen.tician.de/pytools": None,
        "https://documen.tician.de/pymbolic": None,
        }
