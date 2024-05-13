import pandas as pd
import pyarrow as pa
import pytest


def test_nest_accessor(test_dataset):
    """test that the nest accessor is correctly tied to columns"""

    # Make sure that nested columns have the accessor available
    assert hasattr(test_dataset.nested, "nest")

    # Make sure we get an attribute error when trying to use the wrong column
    with pytest.raises(AttributeError):
        test_dataset.ra.nest


def test_fields(test_dataset):
    """test the fields accessor property"""
    assert test_dataset.nested.nest.fields == ["t", "flux", "band"]


def test_to_flat(test_dataset):
    """test the to_flat function"""
    flat_ztf = test_dataset.nested.nest.to_flat()

    # check dtypes
    assert flat_ztf.dtypes["t"] == pd.ArrowDtype(pa.float64())
    assert flat_ztf.dtypes["flux"] == pd.ArrowDtype(pa.float64())
    assert flat_ztf.dtypes["band"] == pd.ArrowDtype(pa.large_string())

    # Make sure we retain all rows
    assert len(flat_ztf.loc[1]) == 500

    one_row = flat_ztf.loc[1].compute().iloc[1]
    assert pytest.approx(one_row["t"], 0.01) == 5.4584
    assert pytest.approx(one_row["flux"], 0.01) == 84.1573
    assert one_row["band"] == "r"


def test_to_flat_with_fields(test_dataset):
    """test the to_flat function"""
    flat_ztf = test_dataset.nested.nest.to_flat(fields=["t", "flux"])

    # check dtypes
    assert flat_ztf.dtypes["t"] == pd.ArrowDtype(pa.float64())
    assert flat_ztf.dtypes["flux"] == pd.ArrowDtype(pa.float64())

    # Make sure we retain all rows
    assert len(flat_ztf.loc[1]) == 500

    one_row = flat_ztf.loc[1].compute().iloc[1]
    assert pytest.approx(one_row["t"], 0.01) == 5.4584
    assert pytest.approx(one_row["flux"], 0.01) == 84.1573


def test_to_lists(test_dataset):
    """test the to_lists function"""
    list_ztf = test_dataset.nested.nest.to_lists()

    # check dtypes
    assert list_ztf.dtypes["t"] == pd.ArrowDtype(pa.list_(pa.float64()))
    assert list_ztf.dtypes["flux"] == pd.ArrowDtype(pa.list_(pa.float64()))
    assert list_ztf.dtypes["band"] == pd.ArrowDtype(pa.list_(pa.large_string()))

    # Make sure we have a single row for an id
    assert len(list_ztf.loc[1]) == 1

    # Make sure we retain all rows -- double loc for speed and pandas get_item
    assert len(list_ztf.loc[1].compute().loc[1]["t"]) == 500

    # spot-check values
    assert pytest.approx(list_ztf.loc[1].compute().loc[1]["t"][0], 0.01) == 7.5690279
    assert pytest.approx(list_ztf.loc[1].compute().loc[1]["flux"][0], 0.01) == 79.6886
    assert list_ztf.loc[1].compute().loc[1]["band"][0] == "g"


def test_to_lists_with_fields(test_dataset):
    """test the to_lists function"""
    list_ztf = test_dataset.nested.nest.to_lists(fields=["t", "flux"])

    # check dtypes
    assert list_ztf.dtypes["t"] == pd.ArrowDtype(pa.list_(pa.float64()))
    assert list_ztf.dtypes["flux"] == pd.ArrowDtype(pa.list_(pa.float64()))

    # Make sure we have a single row for an id
    assert len(list_ztf.loc[1]) == 1

    # Make sure we retain all rows -- double loc for speed and pandas get_item
    assert len(list_ztf.loc[1].compute().loc[1]["t"]) == 500

    # spot-check values
    assert pytest.approx(list_ztf.loc[1].compute().loc[1]["t"][0], 0.01) == 7.5690279
    assert pytest.approx(list_ztf.loc[1].compute().loc[1]["flux"][0], 0.01) == 79.6886
