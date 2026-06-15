# *******************************************************************************
# Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

import os
import glob
import shutil
from pathlib import Path
from setuptools import setup
from setuptools.dist import Distribution
from distutils.command.install import install
from distutils.command.clean import clean
from setuptools.command.build_py import build_py
from setuptools.command.build_clib import build_clib

# setuptools >= 70 ships bdist_wheel under setuptools.command; older releases
# only expose it via the standalone `wheel` package. Support both.
try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

try:
    import torch
except ModuleNotFoundError:
    raise RuntimeError("PyTorch not found, please install PyTorch to continue.")


def _check_build_tools() -> None:
    if shutil.which("cmake") is None:
        raise RuntimeError(
            "pace: required build tool not found on PATH: cmake. "
            "Install via pip install -r build_requirements.txt"
        )


try:
    from setuptools_scm import get_version

    PACKAGE_VERSION = get_version(root=".", relative_to=__file__)
except (ImportError, LookupError):
    # Fallback if setuptools_scm is not available or not in a git repo
    PACKAGE_VERSION = "0.0.0+unknown"


# Extract a CMake-compatible version (major.minor.patch only)
# CMake doesn't support PEP 440 local version identifiers
def extract_cmake_compatible_version(version_string):
    """Extract CMake-compatible version from PEP 440 version string."""
    import re

    # Match major.minor.patch at the start of the version
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version_string)
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return "0.0.0"


PACKAGE_CMAKE_VERSION = extract_cmake_compatible_version(PACKAGE_VERSION)

BUILD_TYPE = "Release"

PACKAGE_NAME = "pace"
CPP_PACKAGE_NAME = PACKAGE_NAME + "_cpp"
PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
PACKAGE_CSRC = os.path.join(PACKAGE_DIR, "csrc")
PACKAGE_BUILD_DIR = os.path.join(PACKAGE_DIR, "build")
PACKAGE_BUILD_TYPE_DIR = os.path.join(PACKAGE_BUILD_DIR, BUILD_TYPE)
PACKAGE_INSALL_DIR = os.path.join(PACKAGE_BUILD_TYPE_DIR, PACKAGE_NAME)


class CPPLibBuild(build_clib):
    def run(self):
        _check_build_tools()
        print(torch.__config__.show())
        print("*" * 45 + "\nBuilding CPP library\n" + "*" * 45)
        print(f"PACE Version: {PACKAGE_VERSION}")
        print(f"CMake Version: {PACKAGE_CMAKE_VERSION}")
        if not os.path.exists(PACKAGE_BUILD_DIR):
            os.makedirs(PACKAGE_BUILD_DIR)

        cmake_cmd = "cmake"
        cmake_cmd += f" -B {PACKAGE_BUILD_DIR}"
        cmake_cmd += f" -S {PACKAGE_DIR}"
        cmake_cmd += f" -DCMAKE_BUILD_TYPE={BUILD_TYPE}"
        cmake_cmd += f" -DPACKAGE_NAME={CPP_PACKAGE_NAME}"
        cmake_cmd += f" -DPACKAGE_VERSION={PACKAGE_CMAKE_VERSION}"
        cmake_cmd += f" -DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}"
        cmake_cmd += f" -DCMAKE_INSTALL_PREFIX={PACKAGE_INSALL_DIR}"
        if os.system(cmake_cmd):
            raise RuntimeError("Build failed, please check the trace.")

        nproc = os.cpu_count()
        make_cmd = f"make -C {PACKAGE_BUILD_DIR} -j {nproc}"
        if os.system(make_cmd):
            raise RuntimeError("Build failed, please check the trace.")

        make_install_cmd = f"make -C {PACKAGE_BUILD_DIR} install"
        if os.system(make_install_cmd):
            raise RuntimeError("Build failed, please check the trace.")


cmdclass = {"build_clib": CPPLibBuild}


def get_src_py_and_dst():
    ret = []
    generated_python_files = glob.glob(
        os.path.join(PACKAGE_DIR, PACKAGE_NAME, "**/*.py"), recursive=True
    )
    for src in generated_python_files:
        dst = os.path.join(
            PACKAGE_INSALL_DIR,
            os.path.relpath(src, os.path.join(PACKAGE_DIR, PACKAGE_NAME)),
        )
        dst_path = Path(dst)
        if not dst_path.parent.exists():
            Path(dst_path.parent).mkdir(parents=True, exist_ok=True)
        ret.append((src, dst))
    return ret


class InstallCmd(install, object):
    def finalize_options(self):
        self.build_lib = os.path.relpath(PACKAGE_BUILD_TYPE_DIR)
        return super(InstallCmd, self).finalize_options()


class CleanCmd(clean, object):
    def run(self):
        from distutils.dir_util import remove_tree

        def _remove(path):
            if os.path.exists(path):
                remove_tree(path)

        _remove(os.path.relpath(PACKAGE_BUILD_DIR))
        _remove(os.path.realpath(PACKAGE_NAME + ".egg-info/"))


class PythonPackageBuild(build_py, object):
    def run(self) -> None:
        # Ensure the CMake build has run so libpace_cpp.so exists before we
        # walk the source tree (build_clib used to be chained via build_ext).
        self.run_command("build_clib")
        ret = get_src_py_and_dst()
        for src, dst in ret:
            self.copy_file(src, dst)
        super(PythonPackageBuild, self).finalize_options()


# Force a platform-tagged wheel even though setuptools sees no ext_modules.
# Without this the wheel would be tagged `any` (pure-Python) and would not
# carry libpace_cpp.so's manylinux/linux platform constraint.
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


# Override the wheel tag to `py3-none-<plat>`: the wheel ships no CPython
# extension (libpace_cpp.so links only torch_cpu+c10, no Python C API), so
# it is CPython-version-agnostic. The minimum Python version is enforced
# by `requires-python` in pyproject.toml, not by the wheel tag.
class BdistWheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return ("py3", "none", plat)


cmdclass["install"] = InstallCmd
cmdclass["build_py"] = PythonPackageBuild
cmdclass["bdist_wheel"] = BdistWheel
cmdclass["clean"] = CleanCmd


setup(
    # `name` is omitted here; pyproject.toml's [project] name = "amd-pace" is
    # authoritative for the distribution name. PACKAGE_NAME stays "pace" for
    # the Python import path / package_data globs.
    packages=[PACKAGE_NAME],
    package_data={PACKAGE_NAME: ["lib/*.so"]},
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass=cmdclass,
)
