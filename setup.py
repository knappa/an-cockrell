#!/usr/bin/env python

import os

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup


def prerelease_local_scheme(version):
    """Return local scheme version unless building on master in Gitlab.

    This function returns the local scheme version number
    (e.g. 0.0.0.dev<N>+g<HASH>) unless building on Gitlab for a
    pre-release in which case it ignores the hash and produces a
    PEP440 compliant pre-release version number (e.g. 0.0.0.dev<N>).
    """
    from setuptools_scm.version import get_local_node_and_date

    if "CIRCLE_BRANCH" in os.environ and os.environ["CIRCLE_BRANCH"] == "master":
        return ""
    else:
        return get_local_node_and_date(version)


setup(
    ext_modules=cythonize("an_cockrell/util.py"),
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    use_scm_version={"local_scheme": prerelease_local_scheme},
)
