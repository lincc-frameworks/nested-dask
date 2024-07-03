from __future__ import annotations

import os

import dask.dataframe as dd
import dask_expr as dx
import nested_pandas as npd
from dask_expr._collection import new_collection
from nested_pandas.series.dtype import NestedDtype
from nested_pandas.series.packer import pack_flat
from pandas._libs import lib
from pandas._typing import AnyAll, Axis, IndexLabel
from pandas.api.extensions import no_default

# need this for the base _Frame class
# mypy: disable-error-code="misc"


class _Frame(dx.FrameBase):  # type: ignore
    """Base class for extensions of Dask Dataframes."""

    _partition_type = npd.NestedFrame

    def __init__(self, expr):
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

    def _rebuild(self, graph, func, args):  # type: ignore
        collection = func(graph, *args)
        return collection


class NestedFrame(
    _Frame, dd.DataFrame
):  # can use dd.DataFrame instead of dx.DataFrame if the config is set true (default in >=2024.3.0)
    """An extension for a Dask Dataframe that has Nested-Pandas functionality.

    Examples
    --------
    >>> import nested_dask as nd
    >>> base = nd.NestedFrame(base_data)
    >>> layer = nd.NestedFrame(layer_data)
    >>> base.add_nested(layer, "layer")
    """

    _partition_type = npd.NestedFrame  # Tracks the underlying data type

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return result

    @classmethod
    def from_pandas(
        cls,
        data,
        npartitions=None,
        chunksize=None,
        sort=True,
    ) -> NestedFrame:
        """Returns an Nested-Dask NestedFrame constructed from a Nested-Pandas
        NestedFrame or Pandas DataFrame.

        Parameters
        ----------
        data: `NestedFrame` or `DataFrame`
            Nested-Pandas NestedFrame containing the underlying data
        npartitions: `int`, optional
            The number of partitions of the index to create. Note that depending on
            the size and index of the dataframe, the output may have fewer
            partitions than requested.
        chunksize: `int`, optional
            The desired number of rows per index partition to use. Note that
            depending on the size and index of the dataframe, actual partition
            sizes may vary.
        sort: `bool`, optional
            Whether to sort the frame by a default index.

        Returns
        ----------
        result: `NestedFrame`
            The constructed Dask-Nested NestedFrame object.
        """
        result = dd.from_pandas(data, npartitions=npartitions, chunksize=chunksize, sort=sort)
        return NestedFrame.from_dask_dataframe(result)

    @classmethod
    def from_dask_dataframe(cls, df: dd.DataFrame) -> NestedFrame:
        """Converts a Dask Dataframe to a Dask-Nested NestedFrame

        Parameters
        ----------
        df:
            A Dask Dataframe to convert

        Returns
        -------
        `nested_dask.NestedFrame`
        """
        return df.map_partitions(npd.NestedFrame, meta=npd.NestedFrame(df._meta.copy()))

    @classmethod
    def from_delayed(cls, dfs, meta=None, divisions=None, prefix="from-delayed", verify_meta=True):
        """
        Create Nested-Dask NestedFrames from many Dask Delayed objects.

        Docstring is copied from `dask.dataframe.from_delayed`.

        Parameters
        ----------
        dfs :
            A ``dask.delayed.Delayed``, a ``distributed.Future``, or an iterable of either
            of these objects, e.g. returned by ``client.submit``. These comprise the
            individual partitions of the resulting dataframe.
            If a single object is provided (not an iterable), then the resulting dataframe
            will have only one partition.
        meta:
            An empty NestedFrame, pd.DataFrame, or pd.Series that matches the dtypes and column names of
            the output. This metadata is necessary for many algorithms in dask dataframe
            to work. For ease of use, some alternative inputs are also available. Instead of a
            DataFrame, a dict of {name: dtype} or iterable of (name, dtype) can be provided (note that
            the order of the names should match the order of the columns). Instead of a series, a tuple of
            (name, dtype) can be used. If not provided, dask will try to infer the metadata. This may lead
            to unexpected results, so providing meta is recommended. For more information, see
            dask.dataframe.utils.make_meta.
        divisions :
            Partition boundaries along the index.
            For tuple, see https://docs.dask.org/en/latest/dataframe-design.html#partitions
            For string 'sorted' will compute the delayed values to find index
            values.  Assumes that the indexes are mutually sorted.
            If None, then won't use index information
        prefix :
            Prefix to prepend to the keys.
        verify_meta :
            If True check that the partitions have consistent metadata, defaults to True.

        """
        nf = dd.from_delayed(dfs=dfs, meta=meta, divisions=divisions, prefix=prefix, verify_meta=verify_meta)
        return NestedFrame.from_dask_dataframe(nf)

    @classmethod
    def from_map(
        cls,
        func,
        *iterables,
        args=None,
        meta=None,
        divisions=None,
        label=None,
        enforce_metadata=True,
        **kwargs,
    ):
        """
        Create a DataFrame collection from a custom function map

        WARNING: The ``from_map`` API is experimental, and stability is not
        yet guaranteed. Use at your own risk!

        Parameters
        ----------
        func : callable
            Function used to create each partition. If ``func`` satisfies the
            ``DataFrameIOFunction`` protocol, column projection will be enabled.
        *iterables : Iterable objects
            Iterable objects to map to each output partition. All iterables must
            be the same length. This length determines the number of partitions
            in the output collection (only one element of each iterable will
            be passed to ``func`` for each partition).
        args : list or tuple, optional
            Positional arguments to broadcast to each output partition. Note
            that these arguments will always be passed to ``func`` after the
            ``iterables`` positional arguments.
        meta:
            An empty NestedFrame, pd.DataFrame, or pd.Series that matches the dtypes and column names of
            the output. This metadata is necessary for many algorithms in dask dataframe
            to work. For ease of use, some alternative inputs are also available. Instead of a
            DataFrame, a dict of {name: dtype} or iterable of (name, dtype) can be provided (note that
            the order of the names should match the order of the columns). Instead of a series, a tuple of
            (name, dtype) can be used. If not provided, dask will try to infer the metadata. This may lead
            to unexpected results, so providing meta is recommended. For more information, see
            dask.dataframe.utils.make_meta.
        divisions : tuple, str, optional
            Partition boundaries along the index.
            For tuple, see https://docs.dask.org/en/latest/dataframe-design.html#partitions
            For string 'sorted' will compute the delayed values to find index
            values.  Assumes that the indexes are mutually sorted.
            If None, then won't use index information
        label : str, optional
            String to use as the function-name label in the output
            collection-key names.
        enforce_metadata : bool, default True
            Whether to enforce at runtime that the structure of the DataFrame
            produced by ``func`` actually matches the structure of ``meta``.
            This will rename and reorder columns for each partition,
            and will raise an error if this doesn't work,
            but it won't raise if dtypes don't match.
        **kwargs:
            Key-word arguments to broadcast to each output partition. These
            same arguments will be passed to ``func`` for every output partition.
        """
        nf = dd.from_map(
            func,
            *iterables,
            args=args,
            meta=meta,
            divisions=divisions,
            label=label,
            enforce_metadata=enforce_metadata,
            **kwargs,
        )
        return NestedFrame.from_dask_dataframe(nf)

    def compute(self, **kwargs):
        """Compute this Dask collection, returning the underlying dataframe or series."""
        return npd.NestedFrame(super().compute(**kwargs))

    @property
    def all_columns(self) -> dict:
        """returns a dictionary of columns for each base/nested dataframe"""
        all_columns = {"base": self.columns}
        for column in self.columns:
            if isinstance(self[column].dtype, NestedDtype):
                nest_cols = list(self.dtypes[column].fields.keys())
                all_columns[column] = nest_cols
        return all_columns

    @property
    def nested_columns(self) -> list:
        """retrieves the base column names for all nested dataframes"""
        nest_cols = []
        for column in self.columns:
            if isinstance(self[column].dtype, NestedDtype):
                nest_cols.append(column)
        return nest_cols

    def add_nested(self, nested, name, how="outer") -> NestedFrame:  # type: ignore[name-defined] # noqa: F821
        """Packs a dataframe into a nested column

        Parameters
        ----------
        nested:
            A flat dataframe to pack into a nested column
        name:
            The name given to the nested column
        how: {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}, default ‘outer’
            How to handle the operation of the two objects.

            * left: use calling frame’s index (or column if on is specified)

            * right: use other’s index.

            * outer: form union of calling frame’s index (or column if on is
            specified) with other’s index, and sort it lexicographically.

            * inner: form intersection of calling frame’s index (or column if
            on is specified) with other’s index, preserving the order of the
            calling’s one.

            * cross: creates the cartesian product from both frames, preserves
            the order of the left keys.

        Returns
        -------
        `nested_dask.NestedFrame`
        """
        nested = nested.map_partitions(lambda x: pack_flat(npd.NestedFrame(x))).rename(name)
        return self.join(nested, how=how)

    def query(self, expr) -> Self:  # type: ignore # noqa: F821:
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
        return self.map_partitions(lambda x: npd.NestedFrame(x).query(expr), meta=self._meta)

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
    ) -> Self:  # type: ignore[name-defined] # noqa: F821:
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
        # propagate meta, assumes row-based operation
        return self.map_partitions(
            lambda x: npd.NestedFrame(x).dropna(
                axis=axis,
                how=how,
                thresh=thresh,
                on_nested=on_nested,
                subset=subset,
                inplace=inplace,
                ignore_index=ignore_index,
            ),
            meta=self._meta,
        )

    def reduce(self, func, *args, meta=None, **kwargs) -> NestedFrame:
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
            columns to apply the function to. See the Notes for recommendations
            on writing func outputs.
        args : positional arguments
            Positional arguments to pass to the function, the first *args should be the names of the
            columns to apply the function to.
        meta : dataframe or series-like, optional
            The dask meta of the output.
        kwargs : keyword arguments, optional
            Keyword arguments to pass to the function.

        Returns
        -------
        `NestedFrame`
            `NestedFrame` with the results of the function applied to the columns of the frame.

        Notes
        -----
        By default, `reduce` will produce a `NestedFrame` with enumerated
        column names for each returned value of the function. For more useful
        naming, it's recommended to have `func` return a dictionary where each
        key is an output column of the dataframe returned by `reduce`.

        Example User Function:

        >>> def my_sum(col1, col2):
        >>>    '''reduce will return a NestedFrame with two columns'''
        >>>    return {"sum_col1": sum(col1), "sum_col2": sum(col2)}

        """

        # apply nested_pandas reduce via map_partitions
        # wrap the partition in a npd.NestedFrame call for:
        # https://github.com/lincc-frameworks/nested-dask/issues/21
        return self.map_partitions(lambda x: npd.NestedFrame(x).reduce(func, *args, **kwargs), meta=meta)

    def to_parquet(self, path, by_layer=True, **kwargs) -> None:
        """Creates parquet file(s) with the data of a NestedFrame, either
        as a single parquet file directory where each nested dataset is packed
        into its own column or as an individual parquet file directory for each
        layer.

        Docstring copied from nested-pandas.

        Note that here we always opt to use the pyarrow engine for writing
        parquet files.

        Parameters
        ----------
        path : str
            The path to the parquet directory to be written.
        by_layer : bool, default True
            NOTE: by_layer=False will not reliably preserve divisions currently,
            be warned when using it that loading from such a dataset will
            likely require you to reset and set the index to generate divisions
            information.

            If False, writes the entire NestedFrame to a single parquet
            directory.

            If True, writes each layer to a separate parquet sub-directory
            within the directory specified by path. The filename for each
            outputted file will be named after its layer. For example for the
            base layer this is always "base".
        kwargs : keyword arguments, optional
            Keyword arguments to pass to the function.

        Returns
        -------
        None
        """

        # code copied from nested-pandas rather than wrapped
        # reason being that a map_partitions call is probably not well-behaved here?

        if not by_layer:
            # Todo: Investigate this more
            # Divisions cannot be generated from a parquet file that stores
            # nested information without a reset_index().set_index() loop. It
            # seems like this happens at the to_parquet level rather than
            # in read_parquet as dropping the nested columns from the dataframe
            # to save does enable divisions to be found, but removing the
            # nested columns from the set of columns to load does not.
            # Divisions are going to be crucial, and so I think it's best to
            # not support this until this is resolved. However the non-by_layer
            # mode is needed for by_layer so it may be best to just settle for
            # changing the default and filing a higher-priority bug.
            # raise NotImplementedError

            # We just defer to the pandas to_parquet method if we're not writing by layer
            # or there is only one layer in the NestedFrame.
            super().to_parquet(path, engine="pyarrow", **kwargs)
        else:
            # Write the base layer to a parquet file
            base_frame = self.drop(columns=self.nested_columns)
            base_frame.to_parquet(os.path.join(path, "base"), by_layer=False, **kwargs)

            # Write each nested layer to a parquet file
            for layer in self.all_columns:
                if layer != "base":
                    path_layer = os.path.join(path, f"{layer}")
                    self[layer].nest.to_flat().to_parquet(path_layer, engine="pyarrow", **kwargs)
        return None
