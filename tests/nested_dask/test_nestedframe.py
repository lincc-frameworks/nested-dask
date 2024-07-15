import dask.dataframe as dd
import nested_dask as nd
import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest
from nested_pandas.series.dtype import NestedDtype


def test_nestedframe_construction(test_dataset):
    """test the construction of a nestedframe"""
    assert len(test_dataset) == 50
    assert test_dataset.columns.to_list() == ["a", "b", "nested"]
    assert isinstance(test_dataset["nested"].dtype, NestedDtype)


def test_nestedframe_from_dask_keeps_index_name():
    """test index name is set in from_dask_dataframe"""
    index_name = "test"
    a = pd.DataFrame({"a": [1, 2, 3]})
    a.index.name = index_name
    ddf = dd.from_pandas(a)
    assert ddf.index.name == index_name
    ndf = nd.NestedFrame.from_dask_dataframe(ddf)
    assert isinstance(ndf, nd.NestedFrame)
    assert ndf.index.name == index_name


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
    assert isinstance(base_with_nested, nd.NestedFrame)

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

    flat_nested_nan_free = nan_free_nested.map_partitions(lambda x: x.nested.nest.to_flat(), meta=meta)
    flat_nested = test_dataset_with_nans.map_partitions(lambda x: x.nested.nest.to_flat(), meta=meta)
    # should just remove one row
    assert len(flat_nested_nan_free) == len(flat_nested) - 1


def test_reduce(test_dataset):
    """test the reduce function"""

    def reflect_inputs(*args):
        return args

    res = test_dataset.reduce(reflect_inputs, "a", "nested.t", meta={0: float, 1: float})

    assert len(res) == 50
    assert isinstance(res.compute().loc[0][0], float)
    assert isinstance(res.compute().loc[0][1], np.ndarray)

    res2 = test_dataset.reduce(np.mean, "nested.flux", meta={0: float})

    assert pytest.approx(res2.compute()[0][15], 0.1) == 53.635174
    assert pytest.approx(sum(res2.compute()[0]), 0.1) == 2488.960119


def test_to_parquet_combined(test_dataset, tmp_path):
    """test to_parquet when saving all layers to a single directory"""

    test_save_path = tmp_path / "test_dataset"

    # send to parquet
    test_dataset.to_parquet(test_save_path, by_layer=False)

    # load back from parquet
    loaded_dataset = nd.read_parquet(test_save_path, calculate_divisions=True)
    # todo: file bug for this and investigate
    loaded_dataset = loaded_dataset.reset_index().set_index("index")

    # Check for equivalence
    assert test_dataset.divisions == loaded_dataset.divisions

    test_dataset = test_dataset.compute()
    loaded_dataset = loaded_dataset.compute()

    assert test_dataset.equals(loaded_dataset)


def test_to_parquet_by_layer(test_dataset, tmp_path):
    """test to_parquet when saving layers to subdirectories"""

    test_save_path = tmp_path / "test_dataset"

    # send to parquet
    test_dataset.to_parquet(test_save_path, by_layer=True, write_index=True)

    # load back from parquet
    loaded_base = nd.read_parquet(test_save_path / "base", calculate_divisions=True)
    loaded_nested = nd.read_parquet(test_save_path / "nested", calculate_divisions=True)

    loaded_dataset = loaded_base.add_nested(loaded_nested, "nested")

    # Check for equivalence
    assert test_dataset.divisions == loaded_dataset.divisions

    test_dataset = test_dataset.compute()
    loaded_dataset = loaded_dataset.compute()

    assert test_dataset.equals(loaded_dataset)


def test_from_epyc():
    """test a dataset from epyc. Motivated by https://github.com/lincc-frameworks/nested-dask/issues/21"""
    # Load some ZTF data
    catalogs_dir = "https://epyc.astro.washington.edu/~lincc-frameworks/half_degree_surveys/ztf/"

    object_ndf = (
        nd.read_parquet(f"{catalogs_dir}/ztf_object", columns=["ra", "dec", "ps1_objid"])
        .set_index("ps1_objid", sort=True)
        .persist()
    )

    source_ndf = (
        nd.read_parquet(
            f"{catalogs_dir}/ztf_source", columns=["mjd", "mag", "magerr", "band", "ps1_objid", "catflags"]
        )
        .set_index("ps1_objid", sort=True)
        .persist()
    )

    object_ndf = object_ndf.add_nested(source_ndf, "ztf_source")

    # Apply a mean function
    meta = pd.DataFrame(columns=[0], dtype=float)
    result = object_ndf.reduce(np.mean, "ztf_source.mag", meta=meta).compute()

    # just make sure the result was successfully computed
    assert len(result) == 9817


@pytest.mark.parametrize("pkg", ["pandas", "nested-pandas"])
@pytest.mark.parametrize("with_nested", [True, False])
def test_from_pandas(pkg, with_nested):
    """Test that from_pandas returns a NestedFrame"""

    if pkg == "pandas":
        df = pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 3])
    elif pkg == "nested-pandas":
        df = npd.NestedFrame({"a": [1, 2, 3]}, index=[1, 2, 3])
        if with_nested:
            nested = npd.NestedFrame({"b": [5, 10, 15, 20, 25, 30]}, index=[1, 1, 2, 2, 3, 3])
            df = df.add_nested(nested, "nested")

    ndf = nd.NestedFrame.from_pandas(df)
    assert isinstance(ndf, nd.NestedFrame)


@pytest.mark.parametrize("with_nested", [True, False])
def test_from_delayed(with_nested):
    """Test that from_delayed returns a NestedFrame"""

    nf = nd.datasets.generate_data(10, 10)
    if not with_nested:
        nf = nf.drop("nested", axis=1)

    delayed = nf.to_delayed()

    ndf = nd.NestedFrame.from_delayed(dfs=delayed, meta=nf._meta)
    assert isinstance(ndf, nd.NestedFrame)


def test_from_map(test_dataset, tmp_path):
    """Test that from_map returns a NestedFrame"""

    # Setup a temporary directory for files
    test_save_path = tmp_path / "test_dataset"

    # Save Base to Parquet
    test_dataset[["a", "b"]].to_parquet(test_save_path, write_index=True)

    # Load from_map
    paths = [
        tmp_path / "test_dataset" / "0.parquet",
        tmp_path / "test_dataset" / "1.parquet",
        tmp_path / "test_dataset" / "2.parquet",
    ]
    ndf = nd.NestedFrame.from_map(nd.read_parquet, paths, meta=test_dataset[["a", "b"]]._meta)
    assert isinstance(ndf, nd.NestedFrame)
