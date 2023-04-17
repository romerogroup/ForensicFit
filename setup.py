from distutils.core import setup
from setuptools import find_packages
import json
import pathlib
import os
from forensicfit import __version__ as version

{
  "name": "forensicfit",
  "version": "1.00",
  "description": "A Python library for forensic image comarision.",
  "author": "Pedram Tavadze, Meghan Prusinowski, Zachary Andrews, Colton Diges, Tatiana Trejos, Aldo H Romero ",
  "email": "petavazohi@mix.wvu.edu, mnp0006@mix.wvu.edu, tatiana.trejos@mail.wvu.edu, alromero@mail.wvu.edu, ",
  "url": "https://github.com/romerogroup/ForensicFit",
  "download_url": "https://github.com/romerogroup/ForensicFit",
  "status": "development",
  "copyright": "Copyright 2021",
  "date": "Jun 16th, 2021"
}

setup(
    name="forensicfit",
    version=version,
    description=data["description"],
    author=data["author"],
    author_email=data["email"],
    url=data["url"],
    download_url=data["download_url"],
    license="LICENSE.txt",
    install_requires=[
        "opencv-python",
        "numpy>=1.20",
        "scipy",
        "matplotlib",
    ],
    data_files=[("", ["LICENSE.txt"])],
    package_data={"": ["setup.json"]},
    scripts=["scripts/create_metadata.py", 
             'scripts/preprocess_bin_based.py', 
             'scripts/store_on_db.py'],
    packages=find_packages(exclude=[".ML-convlolutional_net", "docs"]),
)
