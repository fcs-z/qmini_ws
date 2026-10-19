"""Installation script for the qmini_lab external Isaac Lab extension."""

from pathlib import Path

import toml
from setuptools import find_packages, setup


EXTENSION_ROOT = Path(__file__).resolve().parent
EXTENSION_DATA = toml.load(EXTENSION_ROOT / "config" / "extension.toml")

setup(
    name="qmini_lab",
    version=EXTENSION_DATA["package"]["version"],
    description=EXTENSION_DATA["package"]["description"],
    author=EXTENSION_DATA["package"]["author"],
    maintainer=EXTENSION_DATA["package"]["maintainer"],
    url=EXTENSION_DATA["package"]["repository"],
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    package_data={"qmini_lab": ["assets/qmini/urdf/*.urdf", "assets/qmini/meshes/*.STL"]},
    install_requires=["toml>=0.10.2,<1"],
    python_requires=">=3.12,<3.13",
    classifiers=[
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.0.0",
    ],
    zip_safe=False,
)

