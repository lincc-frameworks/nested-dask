import dask_nested as dn
import numpy as np
import pytest
from nested_pandas.series.dtype import NestedDtype


def test_nestedframe_construction(test_dataset):
    """test the construction of a nestedframe"""
    assert len(test_dataset) == 50
    assert test_dataset.columns.to_list() == ["a", "b", "nested"]
    assert isinstance(test_dataset["nested"].dtype, NestedDtype)


def test_all_columns(test_dataset):
    """all_columns property test"""
    all_cols = test_dataset.all_columns

    assert all_cols["base"].to_list() == test_dataset.columns.to_list()
    assert all_cols["nested"] == ["t", "flux", "band"]


def test_nested_columns(test_dataset):
    """nested_columns property test"""
    assert test_dataset.nested_columns == ["nested"]


def test_add_nested(test_dataset_no_add_nested):
    """test the add_nested function"""
    base, layer = test_dataset_no_add_nested

    base_with_nested = base.add_nested(layer, "nested")

    # Check that the result is a nestedframe
    assert isinstance(base_with_nested, dn.NestedFrame)

    # Check that there's a new nested column with the correct dtype
    assert "nested" in base_with_nested.columns
    assert isinstance(base_with_nested.dtypes["nested"], NestedDtype)

    # Check that the nested partitions were used
    assert base_with_nested.npartitions == 10

    assert len(base_with_nested.compute()) == 50


def test_query_on_base(test_dataset):
    """test the query function on base columns"""

    # Try a few basic queries
    assert len(test_dataset.query("a  > 0.5").compute()) == 22
    assert len(test_dataset.query("a  > 0.5 & b > 1").compute()) == 13
    assert len(test_dataset.query("a  > 2").compute()) == 0


def test_query_on_nested(test_dataset):
    """test the query function on nested columns"""

    # Try a few nested queries
    res = test_dataset.query("nested.flux>75").compute()
    assert len(res.loc[1]["nested"]) == 127

    res = test_dataset.query("nested.band == 'g'").compute()

    assert len(res.loc[1]["nested"]) == 232
    assert len(res) == 50  # make sure the base df remains unchanged


def test_dropna(test_dataset_with_nans):
    """test the dropna function"""

    nan_free_base = test_dataset_with_nans.dropna(subset=["a"])
    # should just remove one row
    assert len(nan_free_base) == len(test_dataset_with_nans) - 1

    meta = test_dataset_with_nans.loc[0].head(0).nested.nest.to_flat()

    nan_free_nested = test_dataset_with_nans.dropna(subset=["nested.t"])

    # import pdb;pdb.set_trace()
    flat_nested_nan_free = nan_free_nested.map_partitions(lambda x: x.nested.nest.to_flat(), meta=meta)
    flat_nested = test_dataset_with_nans.map_partitions(lambda x: x.nested.nest.to_flat(), meta=meta)
    # should just remove one row
    assert len(flat_nested_nan_free) == len(flat_nested) - 1


def test_reduce(test_dataset):
    """test the reduce function"""

    def reflect_inputs(*args):
        return args

    res = test_dataset.reduce(reflect_inputs, "a", "nested.t", meta=("inputs", float))

    assert len(res) == 50
    assert isinstance(res.compute().loc[0][0], float)
    assert isinstance(res.compute().loc[0][1], np.ndarray)

    res2 = test_dataset.reduce(np.mean, "nested.flux", meta=("mean", float))

    assert pytest.approx(res2.compute()[15], 0.1) == 53.635174
    assert pytest.approx(sum(res2.compute()), 0.1) == 2488.960119
