#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Mustafa Hammood",
    author_email='mustafa@siepic.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Minimal Python module to assist in setting up Tidy3D FDTD simulation on planar nanophotonic devices.",
    entry_points={
        'console_scripts': [
            'gds_tidy3d=gds_tidy3d.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='gds_tidy3d',
    name='gds_tidy3d',
    packages=find_packages(include=['gds_tidy3d', 'gds_tidy3d.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mustafacc/gds_tidy3d',
    version='0.1.0',
    zip_safe=False,
)
