from setuptools import setup, find_packages
from pathlib import Path

readme = (Path(__file__).parent / 'README.md')
long_desc = readme.read_text(encoding='utf-8') if readme.exists() else 'murakami lab modules'

setup(
    name='murakami_lab_modules',
    version='0.2.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.0',
        'pandas>=2.2.0',
        'matplotlib>=3.9.0'
    ],
    python_requires='>=3.8',
    author='Yuya Murakami',
    description='A simple library for machine learning',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/yuya-murakami-su/murakami_lab_modules',
    classifiers=[
        'Programming language :: Python :: 3',
        'Operating System :: OS Independent',
    ]
)