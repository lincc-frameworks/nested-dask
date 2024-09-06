"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

import nested_dask as nd
import nested_pandas as npd
import numpy as np
import pandas as pd


def _generate_benchmark_data(add_nested=True):
    """generate a dataset for benchmarks"""

    n_base = 100
    layer_size = 1000

    # use provided seed, "None" acts as if no seed is provided
    randomstate = np.random.RandomState(seed=1)

    # Generate base data
    base_data = {"a": randomstate.random(n_base), "b": randomstate.random(n_base) * 2}
    base_nf = npd.NestedFrame(data=base_data)

    layer_data = {
        "t": randomstate.random(layer_size * n_base) * 20,
        "flux": randomstate.random(layer_size * n_base) * 100,
        "band": randomstate.choice(["r", "g"], size=layer_size * n_base),
        "index": np.arange(layer_size * n_base) % n_base,
    }
    layer_nf = npd.NestedFrame(data=layer_data).set_index("index").sort_index()

    # Convert to Dask
    base_nf = nd.NestedFrame.from_pandas(base_nf).repartition(npartitions=5)
    layer_nf = nd.NestedFrame.from_pandas(layer_nf).repartition(npartitions=50)

    # Return based on add_nested
    if add_nested:
        base_nf = base_nf.add_nested(layer_nf, "nested")
        return base_nf
    else:
        return base_nf, layer_nf


class NestedFrameAddNested:
    """Benchmark the NestedFrame.add_nested function"""

    n_base = 100
    layer_size = 1000
    base_nf = nd.NestedFrame
    layer_nf = nd.NestedFrame

    def setup(self):
        """Set up the benchmark environment"""
        self.base_nf, self.layer_nf = _generate_benchmark_data(add_nested=False)

    def run(self):
        """Run the benchmark."""
        self.base_nf.add_nested(self.layer_nf, "nested").compute()

    def time_run(self):
        """Benchmark the runtime of adding a nested layer"""
        self.run()

    def peakmem_run(self):
        """Benchmark the memory usage of adding a nested layer"""
        self.run()


class NestedFrameReduce:
    """Benchmark the NestedFrame.reduce function"""

    nf = nd.NestedFrame

    def setup(self):
        """Set up the benchmark environment"""
        self.nf = _generate_benchmark_data(add_nested=True)

    def run(self):
        """Run the benchmark."""
        meta = pd.DataFrame(columns=[0], dtype=float)
        self.nf.reduce(np.mean, "nested.flux", meta=meta).compute()

    def time_run(self):
        """Benchmark the runtime of applying the reduce function"""
        self.run()

    def peakmem_run(self):
        """Benchmark the memory usage of applying the reduce function"""
        self.run()


class NestedFrameQuery:
    """Benchmark the NestedFrame.query function"""

    nf = nd.NestedFrame

    def setup(self):
        """Set up the benchmark environment"""
        self.nf = _generate_benchmark_data(add_nested=True)

    def run(self):
        """Run the benchmark."""

        # Apply nested layer query
        self.nf = self.nf.query("nested.band == 'g'").compute()

    def time_run(self):
        """Benchmark the runtime of applying the two queries"""
        self.run()

    def peakmem_run(self):
        """Benchmark the memory usage of applying the two queries"""
        self.run()
