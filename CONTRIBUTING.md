# Welcome to modepy!

## Installation

The latest version of `modepy` is available on PyPI. It can be installed with
```sh
pip install modepy
```

### Obtaining the source

For development, we recommend cloning the [Git
repository](https://github.com/inducer/modepy) and setting up a development
environment. The source can be obtained from GitHub using, e.g., 
```sh
git clone https://github.com/inducer/modepy
```

### Virtual Environments

`modepy` is a pure Python library that (mainly) depends on `numpy`. However, it
has several development dependencies that will be needed when submitting a PR.
To ensure a development environment that does not pollute your system install,
we recommend using [virtual
environments](https://docs.python.org/3/library/venv.html).

A new virtual environment can be created as follows
```sh
python -m venv .venv
```
and activated using
```sh
source .venv/bin/activate
```

These commands may differ slightly based on your operating system, so for more
information see the [official
documentation](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments).
Using other tools, such as [uv](https://docs.astral.sh/uv/pip/environments/) or
[conda](https://docs.conda.io/projects/conda/en/latest/index.html), is also a
good idea.

### Development ("editable") install

Once a virtual environment has been set up, we recommend performing an [editable
install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)
with all the necessary dependencies.

If using the standard `pip`, you can directly run (in your activated virtual
environment!) the following command
```sh
pip install --verbose --editable .[doc,test]
```

## Bug reports

Reporting any issues should be done in the [GitHub Issue
Tracker](https://github.com/inducer/modepy/issues). When reporting an issue,
include as much information as possible to allow reproducing or pinpointing the
exact cause. For example, consider adding

* A small code snippet that reproduces the issue.
* Library versions (especially `numpy`).
* Environment information (operating system, Python version, etc.).

## Contributing


### Coding style

`modepy` uses [ruff](https://github.com/astral-sh/ruff) to check for coding
style and other issues. In general, the library follows the [PEP
8](https://peps.python.org/pep-0008/) rules as much as possible.

Furthermore, we aim for (close to) full type annotated code. This is checked
with [basedpyright](https://docs.basedpyright.com/latest/).

### Pull Requests

Pull requests are very welcome, of course! If you are trying to develop a
larger feature, we recommend opening an issue first to discuss if it is
appropriate for inclusion and any additional implementation details.

To get your PR merged, we require that
* **All** unit tests pass.
* **All** linting tests must pass (`ruff`, `typos`, and any other future additions).
* New functionality should be documented and tested appropriately.

As type annotations are still a work in progress in a considerable part of the
Python ecosystem, we try to be flexible as well. However, new code should
include type annotations, even if they are not completely correct.

### Documentation

You can find the [official documentation](https://documen.tician.de/modepy) for
`modepy` online. The documentation is generated using the
[Sphinx](https://www.sphinx-doc.org/) generator using the default syntax. It
can be generated locally using
```sh
cd doc
make html
```
and loaded from `_build/html/index.html`.

Additions and corrections are very welcome! If you have a cool non-trivial
example of `modepy` usage, also consider adding it to the `examples` folder.

### License

By contributing, you agree that your contributions will be licensed under the
same license as the project (the MIT license). You should also make sure to add
your name to the files you contributed code to.

## Testing

### Running unit tests

`modepy` uses [pytest](https://docs.pytest.org/en/stable/) for its unit testing
needs. All tests (unit tests and doctests) can be run directly using 
```sh
python -m pytest -v -s modepy
```

New unit tests should be added to a new or existing file in `modepy/test`.

### Linting and type annotations

As mentioned before, `modepy` uses `ruff` and `basedpyright` for static
linting. These tools are configured in the `pyproject.toml` file and can be run
directly without additional options. For example, for linting use
```sh
ruff check
```
and for type checking use
```sh
basedpyright
```

