#!/usr/bin/env python

import os
from setuptools import setup, find_packages


def read(fname):
    """Reads a file's contents as a string.

    Args:
        fname (str): Filename.

    Returns:
        File's contents (str).
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


BASE_URL = "https://github.com/anassinator/pddp"
INSTALL_REQUIRES = [
    "gym==0.10.*",
    "numpy>=1.14.0",
    "six>=1.11.0",
    "torch==0.4.1",
    "tqdm==4.25.*",
]
DEV_REQUIRES = ["yapf==0.22.0"]
SETUP_REQUIRES = ["pytest-runner"]
TESTS_REQUIRES = [
    "pytest",
    "pytest-benchmark",
    "pytest-xdist",
]

# Parse version information.
# yapf: disable
version_info = {}
exec(read("pddp/__version__.py"), version_info)
version = version_info["__version__"]
# yapf: enable

setup(
    name="pddp",
    version=version,
    description="Probabilistic Differential Dynamic Programming library",
    long_description=read("README.rst"),
    author="Anass Al",
    author_email="dev@anassinator.com",
    license="GPLv3",
    url=BASE_URL,
    download_url="{}/tarball/{}".format(BASE_URL, version),
    packages=find_packages(),
    zip_safe=True,
    setup_requires=SETUP_REQUIRES,
    extras_require={"dev": DEV_REQUIRES},
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRES,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ])
