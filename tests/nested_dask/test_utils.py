import nested_dask as nd
import pytest
from nested_pandas.utils import count_nested


@pytest.mark.parametrize("join", [True, False])
@pytest.mark.parametrize("by", [None, "band"])
def test_count_nested(test_dataset, join, by):
    """test the count_nested wrapper"""

    # count_nested functionality is tested on the nested-pandas side
    # let's just make sure the behavior here is identical.

    result_dsk = nd.utils.count_nested(test_dataset, "nested", join=join, by=by).compute()
    result_pd = count_nested(test_dataset.compute(), "nested", join=join, by=by)

    assert result_dsk.equals(result_pd)
