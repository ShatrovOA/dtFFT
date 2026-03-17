import os
import re
from pathlib import Path

from skbuild import setup

try:
    import pybind11
    _pybind11_cmake_dir = pybind11.get_cmake_dir()
except (ImportError, AttributeError):
    _pybind11_cmake_dir = None

def _read_cmake_version() -> str:
    """Extract project version from CMakeLists.txt (single source of truth)."""
    cmake = Path(__file__).parent / "CMakeLists.txt"
    match = re.search(r"project\s*\(\s*\w+\s+VERSION\s+([\d.]+)", cmake.read_text())
    if not match:
        raise RuntimeError("Could not find project VERSION in CMakeLists.txt")
    return match.group(1)


version = _read_cmake_version()

features = os.environ.get("EXTRA_FEATURES", "").lower().split(",")
features = {f.strip() for f in features if f.strip()}

# CUPY_PACKAGE controls which cupy variant is declared as a dependency.
# For CUDA 11.x wheels:  CUPY_PACKAGE=cupy-cuda11x
# For CUDA 12.x wheels:  CUPY_PACKAGE=cupy-cuda12x
# Defaults to generic "cupy" (useful for local dev installs).
cupy_package = os.environ.get("CUPY_PACKAGE", "").strip() or "cupy"

# DTFFT_PACKAGE_NAME lets CI publish multiple variants under different PyPI names
# while keeping the same importable namespace "dtfft".
# Examples: dtfft, dtfft-fftw, dtfft-cuda11x, dtfft-cuda12x
package_name = os.environ.get("DTFFT_PACKAGE_NAME", "dtfft").strip()

is_cuda_build = "cuda" in features

# Also detect CUDA intent from CMAKE_ARGS (e.g. when building from sdist via
# CMAKE_ARGS="-DDTFFT_WITH_CUDA=ON ..." pip install dtfft)
_cmake_args_env = os.environ.get("CMAKE_ARGS", "")
_CUDA_CMAKE_FLAGS = {"DTFFT_WITH_CUDA", "DTFFT_WITH_CUFFT", "DTFFT_WITH_VKFFT"}
if not is_cuda_build:
    import re as _re
    for _flag in _CUDA_CMAKE_FLAGS:
        if _re.search(rf"-D{_flag}\s*=\s*(ON|TRUE|1)\b", _cmake_args_env, _re.IGNORECASE):
            is_cuda_build = True
            break

cmake_args = [
    "-DDTFFT_BUILD_PYTHON_API=ON",
]

if _pybind11_cmake_dir:
    cmake_args.append(f"-Dpybind11_DIR={_pybind11_cmake_dir}")

# Map EXTRA_FEATURES names to CMake options.
# Alternatively, pass CMake flags directly via CMAKE_ARGS env var — scikit-build
# merges it automatically:
#   CMAKE_ARGS="-DDTFFT_WITH_FFTW=ON -DDTFFT_WITH_OPENMP=ON" pip install .
options_map = {
    "fftw": "DTFFT_WITH_FFTW",
    "cuda": "DTFFT_WITH_CUDA",
    "cufft": "DTFFT_WITH_CUFFT",
    "tests": "DTFFT_BUILD_TESTS",
}

for feature, cmake_opt in options_map.items():
    if feature in features:
        cmake_args.append(f"-D{cmake_opt}=ON")

if "tests" in features:
    cmake_args.append("-DCMAKE_BUILD_TYPE=Debug")
else:
    cmake_args.append("-DCMAKE_BUILD_TYPE=Release")


def _build_readme(pkg_name: str, feats: set) -> str:
    """Render the Python README template for a specific package variant."""
    template = (Path(__file__).parent / "src" / "interfaces" / "python" / "README.md").read_text()

    _VARIANT_META = {
        "dtfft": (
            "dtfft — transpose-only CPU build",
            "Provides MPI-parallel data transpositions without any FFT configured.",
        ),
        "dtfft-fftw": (
            "dtfft-fftw — CPU build with FFTW3",
            "Includes the [FFTW3](https://www.fftw.org/) FFT backend for CPU transforms.",
        ),
        "dtfft-cuda11x": (
            "dtfft-cuda11x — CPU + GPU build for CUDA 11.x",
            "GPU-accelerated transforms via cuFFT on NVIDIA GPUs with CUDA 11.x.",
        ),
        "dtfft-cuda12x": (
            "dtfft-cuda12x — CPU + GPU build for CUDA 12.x",
            "GPU-accelerated transforms via cuFFT on NVIDIA GPUs with CUDA 12.x.",
        ),
        "dtfft-cuda13x": (
            "dtfft-cuda13x — CPU + GPU build for CUDA 13.x",
            "GPU-accelerated transforms via cuFFT on NVIDIA GPUs with CUDA 13.x.",
        ),
    }

    # Strip MPI suffix before lookup so dtfft-fftw-openmpi → dtfft-fftw
    _mpi_suffix = ""
    for _suffix in ("-openmpi", "-mpich"):
        if pkg_name.endswith(_suffix):
            _mpi_suffix = _suffix
            break
    base_name = pkg_name.removesuffix(_mpi_suffix)
    title, description = _VARIANT_META.get(base_name, (pkg_name, ""))
    if _mpi_suffix:
        mpi_label = "OpenMPI" if _mpi_suffix == "-openmpi" else "MPICH"
        title = f"{title} [{mpi_label}]"

    # Backends table rows
    rows = [
        ("Transpose (MPI)", "✓"),
        ("FFTW3", "✓" if "fftw" in feats else "—"),
        ("cuFFT", "✓" if "cufft" in feats else "—"),
    ]
    backends_table = "| Backend | Available |\n|---|---|\n"
    backends_table += "\n".join(f"| {b} | {s} |" for b, s in rows)

    # Extra install note and requirements for CUDA builds
    if "cuda" in feats:
        cuda_ver = "11" if "cuda11x" in pkg_name else ("13" if "cuda13x" in pkg_name else "12")
        extra_install_note = (
            f"> **Note**: cuFFT and a CUDA {cuda_ver}.x toolkit must be present on the system.\n"
            f"> CuPy is installed automatically as a dependency."
        )
        extra_requirements = f"- CUDA {cuda_ver}.x toolkit\n- NVIDIA GPU (Volta or newer)"
    elif "fftw" in feats:
        extra_install_note = (
            "> **Note**: FFTW3 shared libraries must be installed on the system (`libfftw3`)."
        )
        extra_requirements = "- FFTW3 (`libfftw3-dev` on Debian/Ubuntu, `fftw` via Homebrew)"
    else:
        extra_install_note = ""
        extra_requirements = ""

    return (
        template.replace("{PACKAGE_TITLE}", title)
        .replace("{PACKAGE_NAME}", pkg_name)
        .replace("{VARIANT_DESCRIPTION}", f"> {description}" if description else "")
        .replace("{BACKENDS_TABLE}", backends_table)
        .replace("{EXTRA_INSTALL_NOTE}", extra_install_note)
        .replace("{EXTRA_REQUIREMENTS}", extra_requirements)
    )


if os.environ.get("DTFFT_SDIST", "").strip():
    long_description = (Path(__file__).parent / "src" / "interfaces" / "python" / "README_sdist.md").read_text()
else:
    long_description = _build_readme(package_name, features)

setup(
    name=package_name,
    version=version,
    description="Python bindings for the dtFFT library — distributed FFT via MPI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Oleg Shatrov",
    author_email="",
    url="https://github.com/ShatrovOA/dtFFT",
    project_urls={
        "Documentation": "https://dtfft.readthedocs.io/latest/index.html",
        "Source": "https://github.com/ShatrovOA/dtFFT",
        "Bug Tracker": "https://github.com/ShatrovOA/dtFFT/issues",
    },
    license="GPL-3.0",
    keywords=["fft", "mpi", "hpc", "parallel", "cuda", "distributed"],
    packages=["dtfft"],
    package_dir={"dtfft": "src/interfaces/python"},
    cmake_install_dir="src/interfaces/python",
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "mpi4py",
        # For CUDA builds cupy is a hard dependency — declare the exact variant
        # so pip enforces it.  CUPY_PACKAGE selects cupy-cuda11x / cupy-cuda12x / cupy.
        *([cupy_package] if is_cuda_build else []),
    ],
    extras_require={},
    package_data={"dtfft": ["*.pyi"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: C++",
        "Programming Language :: Fortran",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    cmake_args=cmake_args,
)
