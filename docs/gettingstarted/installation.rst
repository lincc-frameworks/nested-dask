Installation
============

nested-dask is available to install with pip, using the "nested-dask" package name:

.. code-block:: bash

    % pip install nested-dask


This will grab the latest release version of nested-dask from pip.

Installation from Source
---------------------

In some cases, installation via pip may not be sufficient. In particular, if you're looking to grab the latest
development version of nested-dask, you should instead build 'nested-dask' from source. The following process downloads the 
'nested-dask' source code and installs it and any needed dependencies in a fresh conda environment. 

.. code-block:: bash

    conda create -n nested_dask_env python=3.11
    conda activate nested_dask_env

    git clone https://github.com/lincc-frameworks/nested-dask.git
    cd nested-dask
    pip install .
    pip install '.[dev]'

The ``pip install .[dev]`` command is optional, and installs dependencies needed to run the unit tests and build
the documentation. The latest source version of nested-dask may be less stable than a release, and so we recommend 
running the unit test suite to verify that your local install is performing as expected.
