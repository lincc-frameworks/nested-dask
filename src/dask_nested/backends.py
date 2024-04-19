from dask.dataframe.utils import meta_nonempty
from dask.dataframe.dispatch import make_meta_dispatch, pyarrow_schema_dispatch
from dask.dataframe.backends import _nonempty_index, meta_nonempty_dataframe, _nonempty_series
from dask.dataframe.extensions import make_array_nonempty, make_scalar, register_series_accessor
import dask.dataframe as dd

from dask_expr import get_collection_type

import nested_pandas as npd
from nested_pandas.series.ext_array import NestedExtensionArray
from nested_pandas import NestedDtype, NestSeriesAccessor

import pandas as pd

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
    #return pd.DataFrame() # should it be dd.Dataframe()?
    return NestedExtensionArray._from_sequence([pd.DataFrame()], dtype=dtype)
    #return NestedExtensionArray. #TODO: Figure out

#@register_series_accessor("nest")
#class NestSeriesAccessor(npd.NestSeriesAccessor)


