from setuptools import setup, Extension, find_packages
import pybind11
import os

# Define the C++ extension
cpp_extension = Extension(
    name="EliasFanoDB",
    sources=[
        "cpp_src/eliasFano.cpp",
        "cpp_src/QueryScore.cpp",
        "cpp_src/fp_growth.cpp",
        "cpp_src/serialization.cpp",
        "cpp_src/utils.cpp",
    ],
    include_dirs=["cpp_src", "/opt/homebrew/Cellar/armadillo/12.6.5/include"] + [pybind11.get_include()],
    libraries=["armadillo"],
    library_dirs=["/opt/homebrew/Cellar/armadillo/12.6.5/lib"],
    language="c++",
    extra_compile_args=["-std=c++14"],
)

setup(
    name='scfind',
    version='0.1',
    packages=find_packages(),
    author='Nikolaos Patikasi, Shaokun An',
    author_email='shan12@bwh.harvard.edu',
    description='scfind is a method for searching specific cell types from large single-cell datasets by a query of '
                'gene list. scfind can suggest subqueries score by TF-IDF method. scfind can perform hypergeometric '
                'test which allows the evaluation of marker genes specific to each cell type within a dataset.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24.4',
        'scipy>=1.10.1',
        'statsmodels>=0.14.0',
        'pybind11>=2.11.1',
        'pandas>=2.0.3',
        'anndata>=0.9.2',
        'setuptools>=68.0.0',
        'tqdm>=4.66.1',
        'python-Levenshtein>=0.23.0',
        'h5py>=3.10.0',
        'gensim>=4.3.2',
        'fuzzywuzzy>=0.18.0',
        'rapidfuzz>=3.5.2'
    ],
    ext_modules=[cpp_extension],
)
