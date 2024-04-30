import dask.dataframe as dd
import dask_expr as dx
import nested_pandas as npd
from dask_expr._collection import new_collection
from nested_pandas.series.packer import pack_flat
from pandas._libs import lib
from pandas._typing import AnyAll, Axis, IndexLabel
from pandas.api.extensions import no_default


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

    def add_nested(self, nested, name):
        """Packs a dataframe into a nested column

        Parameters
        ----------
        nested:
            A flat dataframe to pack into a nested column
        name:
            The name given to the nested column

        Returns
        -------
        `dask_nested.NestedFrame`
        """
        nested = nested.map_partitions(lambda x: pack_flat(x)).rename(name)
        return self.join(nested, how="outer")

    def query(self, expr):
        """
        Query the columns of a NestedFrame with a boolean expression. Specified
        queries can target nested columns in addition to the typical column set

        Docstring copied from nested-pandas query

        Parameters
        ----------
        expr : str
            The query string to evaluate.

            Access nested columns using `nested_df.nested_col` (where
            `nested_df` refers to a particular nested dataframe and
            `nested_col` is a column of that nested dataframe).

            You can refer to variables
            in the environment by prefixing them with an '@' character like
            ``@a + b``.

            You can refer to column names that are not valid Python variable names
            by surrounding them in backticks. Thus, column names containing spaces
            or punctuations (besides underscores) or starting with digits must be
            surrounded by backticks. (For example, a column named "Area (cm^2)" would
            be referenced as ```Area (cm^2)```). Column names which are Python keywords
            (like "list", "for", "import", etc) cannot be used.

            For example, if one of your columns is called ``a a`` and you want
            to sum it with ``b``, your query should be ```a a` + b``.

        Returns
        -------
        DataFrame
            DataFrame resulting from the provided query expression.

        Notes
        -----
        Queries that target a particular nested structure return a dataframe
        with rows of that particular nested structure filtered. For example,
        querying the NestedFrame "df" with nested structure "my_nested" as
        below will return all rows of df, but with mynested filtered by the
        condition:

        >>> df.query("mynested.a > 2")
        """
        return self.map_partitions(lambda x: x.query(expr))

    def dropna(
        self,
        *,
        axis: Axis = 0,
        how: AnyAll | lib.NoDefault = no_default,
        thresh: int | lib.NoDefault = no_default,
        on_nested: bool = False,
        subset: IndexLabel | None = None,
        inplace: bool = False,
        ignore_index: bool = False,
    ):
        """
        Remove missing values for one layer of the NestedFrame.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Determine if rows or columns which contain missing values are
            removed.

            * 0, or 'index' : Drop rows which contain missing values.
            * 1, or 'columns' : Drop columns which contain missing value.

            Only a single axis is allowed.

        how : {'any', 'all'}, default 'any'
            Determine if row or column is removed from DataFrame, when we have
            at least one NA or all NA.

            * 'any' : If any NA values are present, drop that row or column.
            * 'all' : If all values are NA, drop that row or column.
        thresh : int, optional
            Require that many non-NA values. Cannot be combined with how.
        on_nested : str or bool, optional
            If not False, applies the call to the nested dataframe in the
            column with label equal to the provided string. If specified,
            the nested dataframe should align with any columns given in
            `subset`.
        subset : column label or sequence of labels, optional
            Labels along other axis to consider, e.g. if you are dropping rows
            these would be a list of columns to include.

            Access nested columns using `nested_df.nested_col` (where
            `nested_df` refers to a particular nested dataframe and
            `nested_col` is a column of that nested dataframe).
        inplace : bool, default False
            Whether to modify the DataFrame rather than creating a new one.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        DataFrame or None
            DataFrame with NA entries dropped from it or None if ``inplace=True``.

        Notes
        -----
        Operations that target a particular nested structure return a dataframe
        with rows of that particular nested structure affected.

        Values for `on_nested` and `subset` should be consistent in pointing
        to a single layer, multi-layer operations are not supported at this
        time.
        """
        # grab meta from head, assumes row-based operation
        meta = self.head(0)
        return self.map_partitions(
            lambda x: x.dropna(
                axis=axis,
                how=how,
                thresh=thresh,
                on_nested=on_nested,
                subset=subset,
                inplace=inplace,
                ignore_index=ignore_index,
            ),
            meta=meta,
        )

    def reduce(self, func, *args, **kwargs):
        """
        Takes a function and applies it to each top-level row of the NestedFrame.

        docstring copied from nested-pandas

        The user may specify which columns the function is applied to, with
        columns from the 'base' layer being passsed to the function as
        scalars and columns from the nested layers being passed as numpy arrays.

        Parameters
        ----------
        func : callable
            Function to apply to each nested dataframe. The first arguments to `func` should be which
            columns to apply the function to.
        args : positional arguments
            Positional arguments to pass to the function, the first *args should be the names of the
            columns to apply the function to.
        kwargs : keyword arguments, optional
            Keyword arguments to pass to the function.

        Returns
        -------
        `NestedFrame`
            `NestedFrame` with the results of the function applied to the columns of the frame.

        Notes
        -----
        The recommend return value of func should be a `pd.Series` where the indices are the names of the
        output columns in the dataframe returned by `reduce`. Note however that in cases where func
        returns a single value there may be a performance benefit to returning the scalar value
        rather than a `pd.Series`.
        """

        # apply nested_pandas reduce via map_partitions
        return self.map_partitions(lambda x: x.reduce(func, *args, **kwargs))
