from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='gravmag',
    version='1.0',
    description='Functions for gravmag map inversion',
    author='KetilH',
    author_email='kehok@equinor.com',
    packages=['gravmag'],  #same as name
    install_requires=[], # avoid reinstalling stuff
    # install_requires=required, #external packages as dependencies
    zip_safe=False,
)