import os
import sys
from setuptools import find_packages, setup
from setuptools.command.install import install


__version__ = '0.1.0'

with open('requirements.txt') as f:
    require_packages = [line[:-1] if line[-1] == '\n' else line for line in f]

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='bgg_playground',
    version=__version__,
    author='Lan Jiang',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=require_packages,
    description='Boardgamegeek dataset playground',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            # 'bert = py_bert.__main__:train',
            # 'bert-vocab = py_bert.dataset.vocab:build',
        ]
    }
)