from distutils.core import setup
from setuptools import find_packages
import json
import pathlib
import os

def get_version_info():
    """ 
    Retrieve version info from setup.json
    Method adopted from PyChemia.
    """

    basepath = pathlib.Path(__file__).parent.absolute()
    rf = open(str(basepath) + os.sep + "setup.json")
    release_data = json.load(rf)
    rf.close()

    return release_data

def write_version_py(filename="forensicfit/version.py"):
    """ 
    Write forensicfit/version.py. Adopted from
    PyChemia.
    """

    versioninfo_string = """
        # THIS FILE IS GENERATED FROM Forensicfit SETUP.PY.
        name = '%(name)s'
        version = '%(version)s'
        description = '%(description)s'
        url = '%(url)s'
        author = '%(author)s'
        email = '%(email)s'
        status = '%(status)s'
        copyright = '%(copyright)s'
        date = '%(date)s'
        """
    release_data = get_version_info()

    a = open(filename, "w")
    try:
        a.write(
            versioninfo_string
            % {
                "name": release_data["name"],
                "version": release_data["version"],
                "description": release_data["description"],
                "url": release_data["url"],
                "author": release_data["author"],
                "email": release_data["email"],
                "status": release_data["status"],
                "copyright": release_data["copyright"],
                "date": release_data["date"],
            }
        )
    finally:
        a.close()
    return release_data

data = write_version_py()

setup(
    name=data["name"],
    version=data["version"],
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
