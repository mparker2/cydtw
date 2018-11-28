from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension(
        "cydtw.dtw",
        ["cydtw/dtw.pyx"]
    )
]


setup(
    name='cydtw',
    version='0.1',
    description=(
        'Cython functions for dynamic time warping'
    ),
    author='Matthew Parker',
    packages=[
        'cydtw',
    ],
    ext_modules=cythonize(extensions),
    install_requires=[
        'Cython',
        'numpy',
    ]
)