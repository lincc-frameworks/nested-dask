import nested_dask as nd
import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest


@pytest.fixture
def test_dataset():
    """create a toy dataset for testing purposes"""
    n_base = 50
    layer_size = 500
    randomstate = np.random.RandomState(seed=1)

    # Generate base data
    base_data = {"a": randomstate.random(n_base), "b": randomstate.random(n_base) * 2}
    base_nf = npd.NestedFrame(data=base_data)

    layer_data = {
        "t": randomstate.random(layer_size * n_base) * 20,
        "flux": randomstate.random(layer_size * n_base) * 100,
        # Ensure pyarrow[string] dtype, not large_string
        # https://github.com/lincc-frameworks/nested-dask/issues/71
        "band": pd.Series(
            randomstate.choice(["r", "g"], size=layer_size * n_base), dtype=pd.ArrowDtype(pa.string())
        ),
        "index": np.arange(layer_size * n_base) % n_base,
    }
    layer_nf = npd.NestedFrame(data=layer_data).set_index("index").sort_index()

    base_nd = nd.NestedFrame.from_pandas(base_nf, npartitions=5)
    layer_nd = nd.NestedFrame.from_pandas(layer_nf, npartitions=10)

    return base_nd.add_nested(layer_nd, "nested")


@pytest.fixture
def test_dataset_with_nans():
    """stop before add_nested"""
    n_base = 50
    layer_size = 500
    randomstate = np.random.RandomState(seed=1)

    # Generate base data
    a = randomstate.random(n_base)
    a[10] = np.nan  # add a nan
    base_data = {"a": a, "b": randomstate.random(n_base) * 2}
    base_nf = npd.NestedFrame(data=base_data)

    t = randomstate.random(layer_size * n_base) * 20
    t[50] = np.nan  # add a nan

    layer_data = {
        "t": t,
        "flux": randomstate.random(layer_size * n_base) * 100,
        "band": randomstate.choice(["r", "g"], size=layer_size * n_base),
        "index": np.arange(layer_size * n_base) % n_base,
    }
    layer_nf = npd.NestedFrame(data=layer_data).set_index("index")

    base_nd = nd.NestedFrame.from_pandas(base_nf, npartitions=5)
    layer_nd = nd.NestedFrame.from_pandas(layer_nf, npartitions=10)

    return base_nd.add_nested(layer_nd, "nested")


@pytest.fixture
def test_dataset_no_add_nested():
    """stop before add_nested"""
    n_base = 50
    layer_size = 500
    randomstate = np.random.RandomState(seed=1)

    # Generate base data
    base_data = {"a": randomstate.random(n_base), "b": randomstate.random(n_base) * 2}
    base_nf = npd.NestedFrame(data=base_data)

    layer_data = {
        "t": randomstate.random(layer_size * n_base) * 20,
        "flux": randomstate.random(layer_size * n_base) * 100,
        "band": randomstate.choice(["r", "g"], size=layer_size * n_base),
        "index": np.arange(layer_size * n_base) % n_base,
    }
    layer_nf = npd.NestedFrame(data=layer_data).set_index("index")

    base_nd = nd.NestedFrame.from_pandas(base_nf, npartitions=5)
    layer_nd = nd.NestedFrame.from_pandas(layer_nf, npartitions=10)

    return (base_nd, layer_nd)
