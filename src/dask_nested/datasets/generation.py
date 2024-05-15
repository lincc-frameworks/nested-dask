from nested_pandas import datasets

import dask_nested as dn


def generate_data(n_base, n_layer, npartitions=1, seed=None) -> dn.NestedFrame:
    """Generates a toy dataset.

    Docstring copied from nested-pandas.

    Parameters
    ----------
    n_base : int
        The number of rows to generate for the base layer
    n_layer : int, or dict
        The number of rows per n_base row to generate for a nested layer.
        Alternatively, a dictionary of layer label, layer_size pairs may be
        specified to created multiple nested columns with custom sizing.
    npartitions: int
        The number of partitions to split the data into.
    seed : int
        A seed to use for random generation of data

    Returns
    -------
    NestedFrame
        The constructed Dask NestedFrame.

    Examples
    --------
    >>> import dask_nested as dn
    >>> dn.datasets.generate_data(10,100)
    >>> dn.datasets.generate_data(10, {"nested_a": 100, "nested_b": 200})
    """

    # Use nested-pandas generator
    base_nf = datasets.generate_data(n_base, n_layer, seed=seed)

    # Convert to dask-nested
    base_nf = dn.NestedFrame.from_nestedpandas(base_nf).repartition(npartitions=npartitions)

    return base_nf
