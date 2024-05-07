import dask_nested as dn
from nested_pandas.series.dtype import NestedDtype


def test_nestedframe_construction(test_dataset):
    """test the construction of a nestedframe"""
    pass


def test_all_columns(test_dataset):
    """all_columns property test"""
    pass


def test_nested_columns(test_dataset):
    """nested_columns property test"""
    pass


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


def test_query(test_dataset):
    """test the query function"""
    pass


def test_dropna(test_dataset):
    """test the dropna function"""
    pass


def test_reduce(test_dataset):
    """test the reduce function"""
    pass
