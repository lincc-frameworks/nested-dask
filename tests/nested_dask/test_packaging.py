import nested_dask


def test_version():
    """Check to see that the version property returns something"""
    assert nested_dask.__version__ is not None
