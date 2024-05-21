from nested_pandas import datasets

import nested_dask as nd


def generate_data(n_base, n_layer, npartitions=1, seed=None) -> nd.NestedFrame:
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
    >>> import nested_dask as nd
    >>> nd.datasets.generate_data(10,100)
    >>> nd.datasets.generate_data(10, {"nested_a": 100, "nested_b": 200})
    """

    # Use nested-pandas generator
    base_nf = datasets.generate_data(n_base, n_layer, seed=seed)

    # Convert to nested-dask
    base_nf = nd.NestedFrame.from_nested_pandas(base_nf).repartition(npartitions=npartitions)

    return base_nf
