import nested_pandas as npd
import pandas as pd
from dask.dataframe.backends import meta_nonempty_dataframe
from dask.dataframe.dispatch import make_meta_dispatch
from dask.dataframe.extensions import make_array_nonempty
from dask.dataframe.utils import meta_nonempty
from dask_expr import get_collection_type
from nested_pandas.series.ext_array import NestedExtensionArray

from .core import NestedFrame

get_collection_type.register(npd.NestedFrame, lambda _: NestedFrame)


@make_meta_dispatch.register(npd.NestedFrame)
def make_meta_frame(x, index=None):
    # Create an empty NestedFrame to use as Dask's underlying object meta.
    result = x.head(0)
    return result


@meta_nonempty.register(npd.NestedFrame)
def _nonempty_nestedframe(x, index=None):
    # Construct a new NestedFrame with the same underlying data.
    df = meta_nonempty_dataframe(x)
    return npd.NestedFrame(df)


@make_array_nonempty.register(npd.NestedDtype)
def _(dtype):
    # must be two values
    return NestedExtensionArray._from_sequence([pd.NA, pd.NA], dtype=dtype)
