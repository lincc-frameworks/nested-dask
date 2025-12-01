# nested-dask

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/nested-dask?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/nested-dask/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/nested-dask.svg?color=blue&logo=condaforge&logoColor=white)](https://anaconda.org/conda-forge/nested-dask)

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/nested-dask/smoke-test.yml)](https://github.com/lincc-frameworks/nested-dask/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/lincc-frameworks/nested-dask/branch/main/graph/badge.svg)](https://codecov.io/gh/lincc-frameworks/nested-dask)
[![Read The Docs](https://img.shields.io/readthedocs/nested-dask)](https://nested-dask.readthedocs.io/)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/nested-dask/asv-main.yml?label=benchmarks)](https://lincc-frameworks.github.io/nested-dask/)

__ARCHIVE NOTICE: This repository is no longer being maintained. The original purpose of this package was to enable dask with the extended nested-pandas API, with the specific aim of using this to support the [LSDB](https://github.com/astronomy-commons/lsdb) project. In April 2025, the contents of this package were [migrated directly into LSDB](https://github.com/astronomy-commons/lsdb/pull/713) to allow for tailored behavior to LSDB's specific operational needs. As a result, any needed changes to dask-compatibility for nested-pandas are happening directly within LSDB and not in this repository. Further, maintaining a generalized dask layer for nested-pandas is not directly within the critical path for LINCC-Frameworks effort at this time. If you found your way here and wished there was a maintained package that provided a dask-layer for nested-pandas for something you are working on, please feel free to voice that as an issue filed to [nested-pandas](https://github.com/lincc-frameworks/nested-pandas/issues).__

A [dask](https://www.dask.org/) extension of 
[nested-pandas](https://nested-pandas.readthedocs.io/en/latest/).

Nested-pandas is a pandas extension package that empowers efficient analysis
of nested associated datasets. This package wraps the majority of the 
nested-pandas API with Dask, which enables easy parallelization and capacity 
for work at scale.



## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create env -n <env_name> python=3.10
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> pip install -e .'[dev]'
>> pre-commit install
>> conda install pandoc
```

Notes:
1. The single quotes around `'[dev]'` may not be required for your operating system.
2. `pre-commit install` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
3. Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)
