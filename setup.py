from setuptools import find_packages, setup
from forensicfit import __version__ as version

with open("requirements.txt") as f:
    requirements = []
    extras = {}
    reqs = f.read().splitlines()
    optional = False
    for req in reqs:
        if len(req) == 0:
            continue
        if req.startswith("#"):
            req = req.replace("#", "").strip()
            if "optional" in req.lower():
                section = req.lower().replace('optional', '').strip()
                extras[section] = []
                optional = True
            else:
                extras[section].append(req)
        else:
            requirements.append(req)

setup(
    name="forensicfit",
    version=version,
    description="A Python library for forensic image comparision.",
    author='Pedram Tavadze',
    author_email='petavazohi@mix.wvu.edu',
    url="https://github.com/romerogroup/ForensicFit",
    license="LICENSE.txt",
    install_requires=requirements,
    extras_require=extras,
    scripts=["scripts/create_metadata.py", 
             'scripts/preprocess_bin_based.py', 
             'scripts/store_on_db.py'],
    packages=find_packages(exclude=["docs", 'sphinx']),
)
