import dask_nested as dn


def test_generate_data():
    """test the dataset generator function"""

    # test the seed
    generate_1 = dn.datasets.generate_data(10, 100, npartitions=2, seed=1)
    generate_2 = dn.datasets.generate_data(10, 100, npartitions=2, seed=1)
    generate_3 = dn.datasets.generate_data(10, 100, npartitions=2, seed=2)

    assert generate_1.compute().equals(generate_2.compute())
    assert not generate_1.compute().equals(generate_3.compute())

    # test npartitions
    assert generate_1.npartitions == 2

    # test the length
    assert len(generate_1) == 10
    assert len(generate_1.nested.nest.to_flat()) == 1000
