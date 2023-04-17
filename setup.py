from distutils.core import setup
from setuptools import find_packages
import json
import pathlib
import os
from forensicfit import __version__ as version

with open("requirements.txt") as f:
    requirements = []
    optionals = []
    reqs = f.read().splitlines()
    for req in reqs:
        if req.startswith("#"):
            continue
        if req.startswith("-"):
            optionals.append(req[1:])
        else:
            requirements.append(req)

setup(
    name="forensicfit",
    version=version,
    description="A Python library for forensic image comparision.",
    author='Pedram Tavadze',
    author_email='petavazohi@mix.wvu.edu',
    url="https://github.com/romerogroup/ForensicFit",
    download_url=data["download_url"],
    license="LICENSE.txt",
    install_requires=[
        "opencv-python",
        "numpy>=1.20",
        "scipy",
        "matplotlib",
    ],
    data_files=[("", ["LICENSE.txt"])],
    scripts=["scripts/create_metadata.py", 
             'scripts/preprocess_bin_based.py', 
             'scripts/store_on_db.py'],
    packages=find_packages(exclude=[".ML-convlolutional_net", "docs"]),
)
