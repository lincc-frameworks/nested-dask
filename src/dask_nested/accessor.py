import nested_pandas as npd
from dask.dataframe.extensions import register_series_accessor
from nested_pandas import NestedDtype


@register_series_accessor("nest")
class DaskNestSeriesAccessor(npd.NestSeriesAccessor):
    """The nested-dask version of the nested-pandas NestSeriesAccessor.

    Note that this has a very limited implementation relative to nested-pandas.

    Parameters
    ----------
    series: dd.series
        A series to tie to the accessor

    """

    def __init__(self, series):
        self._check_series(series)

        self._series = series

    @staticmethod
    def _check_series(series):
        """chcek the validity of the tied series dtype"""
        dtype = series.dtype
        if not isinstance(dtype, NestedDtype):
            raise AttributeError(f"Can only use .nest accessor with a Series of NestedDtype, got {dtype}")

    @property
    def fields(self) -> list[str]:
        """Names of the nested columns"""

        return self._series.head(0).nest.fields  # hacky
