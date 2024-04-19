from dask.dataframe.extensions import make_array_nonempty, make_scalar, register_series_accessor
import nested_pandas as npd


@register_series_accessor("nest")
class DaskNestSeriesAccessor(npd.NestSeriesAccessor):
    pass
