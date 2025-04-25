# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pgmQC',
    version='0.1.0',
    description='',
    long_description=readme,
    author='Zirui Li',
    author_email='zirui.li@rutgers.edu',
    url='https://github.com/abclzr/pgmQC_ae',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'dataset', 'hpca_2025_examples', 'asplos_2025_examples', 'cuquantum_examples', 'sparse_contract_examples', 'tensor_contraction_gpu', 'recursive_nnf', 'shots_assignment', 'artifact'))
)