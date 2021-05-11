from setuptools import setup, Extension
import os
import re


def get_version(package):
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(pwd, package, "__init__.py"), "r") as input:
        result = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', input.read())
    if not result:
        raise ValueError("failed to determine {} version".format(package))
    return result.group(1)


setup(
    name="annealing-sign-problem",
    version=get_version("annealing_sign_problem"),
    description="See README.md",
    url="http://github.com/twesterhout/annealing-sign-problem",
    author="Tom Westerhout",
    author_email="14264576+twesterhout@users.noreply.github.com",
    license="BSD3",
    packages=["annealing_sign_problem"],
    package_data={"annealing_sign_problem": ["libbuild_matrix.so"]},
    install_requires=["numpy", "scipy", "lattice-symmetries", "loguru"],
    zip_safe=False,
)
