"""Build the experimental C++ extension for the installed Linux RDKit wheel.

Use a C++17 compiler and development headers matching the wheel's RDKit and
Boost versions. For this project's pin, conda-forge's librdkit-dev=2025.03.6
and libboost-headers=1.85.0 supply them. Pass the directories containing
GraphMol/ and boost/ respectively. Only headers are taken from this SDK.
"""

import argparse
import os
import shlex
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import rdkit
from rdkit import rdBase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdkit-include", type=Path, required=True)
    parser.add_argument("--boost-include", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This build helper targets the Linux wheel.")
    if rdBase.rdkitVersion != "2025.03.6" or rdBase.boostVersion != "1_85":
        parser.error("This build targets the RDKit 2025.03.6 / Boost 1.85 wheel.")
    if not (args.rdkit_include / "GraphMol/RWMol.h").is_file():
        parser.error("--rdkit-include must contain GraphMol/RWMol.h")
    if not (args.boost_include / "boost/python.hpp").is_file():
        parser.error("--boost-include must contain boost/python.hpp")

    libs = Path(rdkit.__file__).resolve().parents[1] / "rdkit.libs"
    core = Path(__file__).resolve().parents[1] / "moldr/core"
    module = core / ("_native_reassembly" + sysconfig.get_config_var("EXT_SUFFIX"))
    python_abi = f"{sys.version_info.major}{sys.version_info.minor}"
    lib_paths = []
    for name in (
        "RDKitSmilesParse",
        "RDKitGraphMol",
        "RDKitRDGeneral",
        f"boost_python{python_abi}",
    ):
        matches = sorted(libs.glob(f"lib{name}-*.so*"))
        if len(matches) != 1:
            parser.error(
                f"Expected one lib{name} shared library in {libs}, found {matches}"
            )
        lib_paths.append(str(matches[0]))
    with tempfile.TemporaryDirectory(dir=core) as temp:
        output = Path(temp) / module.name
        cmd = [
            *shlex.split(os.environ.get("CXX", "c++")),
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-fPIC",
            "-shared",
            f"-I{sysconfig.get_path('include')}",
            f"-I{args.rdkit_include}",
            f"-I{args.boost_include}",
            str(core / "native_reassembly.cpp"),
            str(core / "native_reassembly_binding.cpp"),
            *lib_paths,
            # The wheel's transitive RDKit dependencies live alongside GraphMol.
            f"-Wl,--disable-new-dtags,-rpath,{libs}",
            "-o",
            str(output),
        ]
        subprocess.run(cmd, check=True)
        output.replace(module)
    print(module)


if __name__ == "__main__":
    main()
