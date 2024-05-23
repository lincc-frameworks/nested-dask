Contribution Guide
==================

Dev Guide - Getting Started
---------------------------

Download code and install dependencies in a conda environment. Run unit tests at the end as a verification that the packages are properly installed.

.. code-block:: bash

    conda create -n nested_dask_env python=3.11
    conda activate nested_dask_env

    git clone https://github.com/lincc-frameworks/nested-dask.git
    cd nested-dask/
    bash ./.setup_dev.sh
