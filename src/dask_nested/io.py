import dask.dataframe as dd

from .core import NestedFrame


def read_parquet(
    path,
    columns=None,
    filters=None,
    categories=None,
    index=None,
    storage_options=None,
    engine="auto",
    use_nullable_dtypes: bool | None = None,
    dtype_backend=None,
    calculate_divisions=None,
    ignore_metadata_file=False,
    metadata_task_size=None,
    split_row_groups="infer",
    blocksize="default",
    aggregate_files=None,
    parquet_file_extension=(".parq", ".parquet", ".pq"),
    filesystem=None,
    **kwargs,
):
    return NestedFrame.from_dask_dataframe(
        dd.read_parquet(
            path=path,
            columns=columns,
            filters=filters,
            categories=categories,
            index=index,
            storage_options=storage_options,
            engine=engine,
            use_nullable_dtypes=use_nullable_dtypes,
            dtype_backend=dtype_backend,
            calculate_divisions=calculate_divisions,
            ignore_metadata_file=ignore_metadata_file,
            metadata_task_size=metadata_task_size,
            split_row_groups=split_row_groups,
            blocksize=blocksize,
            aggregate_files=aggregate_files,
            parquet_file_extension=parquet_file_extension,
            filesystem=filesystem,
            **kwargs,
        )
    )
