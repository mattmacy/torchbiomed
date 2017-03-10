#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages


readme = open('README.md').read()

VERSION = '0.0.1'

# same as ./requirements.txt
requirements = [
    'numpy',
    'torch',
    'torchvision',
]

setup(
    # Metadata
    name='torchbiomed',
    version=VERSION,
    author='Matthew Macy',
    author_email='mat.macy@gmail.com',
    url='https://github.com/mattmacy/torchbiomed',
    description='biomedical image datasets, transforms, utilities, and models for torch deep learning',
    long_description=readme,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),
    zip_safe=True,
    install_requires=requirements,
)
