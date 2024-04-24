from dask.dataframe.extensions import make_array_nonempty, make_scalar, register_series_accessor
import nested_pandas as npd
from nested_pandas import NestedDtype, NestSeriesAccessor
import dask


@register_series_accessor("nest")
class DaskNestSeriesAccessor(npd.NestSeriesAccessor):
    
    def __init__(self, series):
        self._check_series(series)

        self._series = series

    @staticmethod
    def _check_series(series):
        dtype = series.dtype
        if not isinstance(dtype, NestedDtype):
            raise AttributeError(f"Can only use .nest accessor with a Series of NestedDtype, got {dtype}")

    @property
    def fields(self) -> list[str]:
        """Names of the nested columns"""
        return self._series.head(0).nest.fields
        #hacky
        #return self._series.partitions[0:1].map_partitions(lambda x: x.nest.fields)
        #return self._series.array.field_names

    @dask.delayed
    def test_fields(self):
        return self._series.head(0).nest.fields
