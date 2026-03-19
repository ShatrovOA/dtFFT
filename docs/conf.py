# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import enum
import glob
import os
import subprocess
import sys

project = "dtFFT"
copyright = "2025, Oleg Shatrov"
author = "Oleg Shatrov"
release = "latest"

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

try:
    import dtfft  # noqa: F401
except ModuleNotFoundError:
    pattern = os.path.join(root, "_skbuild", "*", "setuptools", "lib.*")
    candidates = sorted(glob.glob(pattern))
    if candidates:
        sys.path.insert(0, candidates[0])
        print("candidates = ", candidates[0], file=sys.stderr)
    import dtfft  # noqa: F401


def _inject_docs_optional_symbols(mod):
    class Platform(enum.Enum):
        """Runtime platform enum used by dtFFT plans."""

        HOST = 0
        CUDA = 1

    class CompressionMode(enum.Enum):
        """Compression mode."""

        def __new__(cls, value, doc: str):
            obj = object.__new__(cls)
            obj._value_ = value
            obj.__doc__ = doc
            return obj

        LOSSLESS = (1, "Lossless compression mode.")
        FIXED_RATE = (2, "Fixed rate compression mode.")
        FIXED_PRECISION = (3, "Fixed precision compression mode.")
        FIXED_ACCURACY = (4, "Fixed accuracy compression mode.")

    class CompressionLib(enum.Enum):
        """Compression backend library."""

        def __new__(cls, value, doc: str):
            obj = object.__new__(cls)
            obj._value_ = value
            obj.__doc__ = doc
            return obj

        ZFP = (1, "Use ZFP compression library.")

    class CompressionConfig:
        """Compression configuration for transpose/reshape paths.

        Parameters
        ----------
        compression_lib : CompressionLib, optional
            Compression library to use. Default is CompressionLib.ZFP.
        compression_mode : CompressionMode, optional
            Compression mode to use. Default is CompressionMode.LOSSLESS.
        rate : float, optional
            Target rate for fixed rate mode. Ignored for other modes. Default is 0.0.
        precision : int, optional
            Target precision for fixed precision mode. Ignored for other modes. Default is 0.0.
        tolerance : float, optional
            Target tolerance for fixed accuracy mode. Ignored for other modes. Default is 0.0.
        """

        def __init__(
            self,
            compression_lib: CompressionLib = CompressionLib.ZFP,
            compression_mode: CompressionMode = CompressionMode.LOSSLESS,
            rate: float = 0.0,
            precision: int = 0,
            tolerance: float = 0.0,
        ): ...

    for cls in (Platform, CompressionMode, CompressionLib, CompressionConfig):
        cls.__module__ = "dtfft"
        cls.__qualname__ = cls.__name__

    if not hasattr(mod, "Platform"):
        mod.Platform = Platform
    if not hasattr(mod, "CompressionMode"):
        mod.CompressionMode = CompressionMode
    if not hasattr(mod, "CompressionLib"):
        mod.CompressionLib = CompressionLib
    if not hasattr(mod, "CompressionConfig"):
        mod.CompressionConfig = CompressionConfig


if os.environ.get("DTFFT_DOCS_FAKE_OPTIONAL", "").lower() in ("1", "true", "yes", "on"):
    _inject_docs_optional_symbols(dtfft)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "breathe",
    "sphinxfortran.fortran_domain",
    "sphinx_tabs.tabs",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "enum_tools.autoenum",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
# autodoc_typehints = 'description'
# autoclass_content = 'init'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "images/logo.png"
html_favicon = "images/favicon.png"

breathe_projects = {"dtFFT": "xml/"}
breathe_default_project = "dtFFT"

read_the_docs_build = os.environ.get("READTHEDOCS", None) == "True"

if read_the_docs_build:
    subprocess.call("doxygen", shell=True)
