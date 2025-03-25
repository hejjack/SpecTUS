from setuptools import setup, find_packages

setup(
    name="spectus",
    version="0.1",
    packages=find_packages(where="spectus"),
    package_dir={"": "spectus"},
)