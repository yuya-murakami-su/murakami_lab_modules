from setuptools import setup, find_packages

setup(
    name='murakami_lab_modules',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy>1.26.0',
        'pandas>2.2.0',
        'torch>2.5.1',
        'matplotlib>3.10.0'
    ],
    author='Yuya Murakami',
    description='A simple library for machine learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yuya-murakami-su/murakami_lab_modules',
    classifiers=[
        'Programming languages :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8'
)