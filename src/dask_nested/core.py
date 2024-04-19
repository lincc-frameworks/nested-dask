import dask
import dask.dataframe as dd
import dask_expr as dx
from dask_expr import get_collection_type
from dask_expr._collection import new_collection, from_dict
from dask_expr._expr import _emulate, ApplyConcatApply

import nested_pandas as npd


class _Frame(dx.FrameBase):
    """Base class for extensions of Dask Dataframes that track additional
    Ensemble-related metadata.
    """

    _partition_type = npd.NestedFrame

    def __init__(self, expr, label=None, ensemble=None):
        super().__init__(expr)

    @property
    def _args(self):
        # Ensure our Dask extension can correctly be used by pickle.
        # See https://github.com/geopandas/dask-geopandas/issues/237
        return super()._args
    
    def optimize(self, fuse: bool = True):
        result = new_collection(self.expr.optimize(fuse=fuse))
        return result

    def __dask_postpersist__(self):
        func, args = super().__dask_postpersist__()

        return self._rebuild, (func, args)

    def _rebuild(self, graph, func, args):
        collection = func(graph, *args)
        return collection


class NestedFrame(
    _Frame, dd.DataFrame
):  # can use dd.DataFrame instead of dx.DataFrame if the config is set true (default in >=2024.3.0)
    """An extension for a Dask Dataframe for Nested.

    The underlying non-parallel dataframes are TapeFrames and TapeSeries which extend Pandas frames.

    Examples
    ----------
    Instatiation::

        import tape
        ens = tape.Ensemble()
        data = {...} # Some data you want tracked by the Ensemble
        ensemble_frame = tape.EnsembleFrame.from_dict(data, label="my_frame", ensemble=ens)
    """

    _partition_type = npd.NestedFrame  # Tracks the underlying data type

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return result

    @classmethod
    def from_nestedpandas(cls, data, npartitions=None, chunksize=None, sort=True, label=None, ensemble=None):
        """Returns an EnsembleFrame constructed from a TapeFrame.

        Parameters
        ----------
        data: `TapeFrame`
            Frame containing the underlying data fro the EnsembleFram
        npartitions: `int`, optional
            The number of partitions of the index to create. Note that depending on
            the size and index of the dataframe, the output may have fewer
            partitions than requested.
        chunksize: `int`, optional
            Size of the individual chunks of data in non-parallel objects that make up Dask frames.
        sort: `bool`, optional
            Whether to sort the frame by a default index.
        label: `str`, optional
            The label used to by the Ensemble to identify the frame.
        ensemble: `tape.Ensemble`, optional
            A link to the Ensemble object that owns this frame.

        Returns
        ----------
        result: `tape.EnsembleFrame`
            The constructed EnsembleFrame object.
        """
        result = dd.from_pandas(data, npartitions=npartitions, chunksize=chunksize, sort=sort)
        return result
    
    @classmethod
    def from_dask_dataframe(cl, df, ensemble=None, label=None):
        """Returns an EnsembleFrame constructed from a Dask dataframe.

        Parameters
        ----------
        df: `dask.dataframe.DataFrame` or `list`
            a Dask dataframe to convert to an EnsembleFrame
        ensemble: `tape.ensemble.Ensemble`, optional
            A link to the Ensemble object that owns this frame.
        label: `str`, optional
            The label used to by the Ensemble to identify the frame.

        Returns
        ----------
        result: `tape.EnsembleFrame`
            The constructed EnsembleFrame object.
        """
        # Create a EnsembleFrame by mapping the partitions to the appropriate meta, TapeFrame
        # TODO(wbeebe@uw.edu): Determine if there is a better method
        result = df.map_partitions(npd.NestedFrame)
        result.ensemble = ensemble
        result.label = label
        return result