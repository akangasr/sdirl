#!/usr/bin/env python

import os
from setuptools import setup

requirements = [
                'elfi',
                'urllib3',
                # 'https://github.com/pybrain/pybrain/archive/0.3.3.zip',
                'numpy>=1.8',
                'scipy>=0.16.1',
                #'gpyopt'
                ]

setup(
    name='SDIRL',
    packages=['sdirl'],
    version='0.1',
    author='Antti Kangasraasio',
    author_email='antti.kangasraasio@iki.fi',
    url='https://github.com/akangasr/sdirl',
    install_requires=requirements,
    description='Summary Data Inverse Reinforcement Learning')
