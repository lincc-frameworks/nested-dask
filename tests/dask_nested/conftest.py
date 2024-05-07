import dask_nested as dn
import nested_pandas as npd
import numpy as np
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
        "band": randomstate.choice(["r", "g"], size=layer_size * n_base),
        "index": np.arange(layer_size * n_base) % n_base,
    }
    layer_nf = npd.NestedFrame(data=layer_data).set_index("index")

    base_dn = dn.NestedFrame.from_nestedpandas(base_nf, npartitions=5)
    layer_dn = dn.NestedFrame.from_nestedpandas(layer_nf, npartitions=10)

    return base_dn.add_nested(layer_dn, "nested")


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

    base_dn = dn.NestedFrame.from_nestedpandas(base_nf, npartitions=5)
    layer_dn = dn.NestedFrame.from_nestedpandas(layer_nf, npartitions=10)

    return (base_dn, layer_dn)
