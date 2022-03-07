import dask.array as da
from sklearn.pipeline import Pipeline
from pymks import (
    GenericTransformer,
    PrimitiveTransformer,
    TwoPointCorrelation,
)
import time
import numpy as np
from dask_ml.decomposition import PCA, IncrementalPCA
from sklearn.decomposition import PCA as PCA_sklearn
import itertools
from dask.distributed import Client
import click


def prepare_data(n_sample, n_chunk):
    x_data = da.from_zarr("../../data/x_data.zarr" , chunks=(100, -1))
    y_data = np.load("../../data/y_data.npy")
    y_data = da.from_array(y_data, chunks=(100, -1))

    # resize data
    x_data = x_data[:n_sample].rechunk((n_sample // n_chunk,) +  x_data.shape[1:])
    y_data = y_data[:n_sample].rechunk((n_sample // n_chunk,) +  y_data.shape[1:])

    pipeline = Pipeline([
        ("reshape", GenericTransformer(
            lambda x: x.reshape(x.shape[0], 51, 51,51)
        )),
        ("discritize",PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0)),
        ("correlations",TwoPointCorrelation(periodic_boundary=True, correlations=[(0, 0)])),
        ('flatten', GenericTransformer(lambda x: x.reshape(x.shape[0], -1)))
    ])

    return pipeline.transform(x_data).persist(), y_data.persist()

def pca_func(x_data, y_data, pca):
    start = time.time()
    try:
        print(x_data.shape)
        print(y_data.shape)
        print(x_data.chunks)
        pca.fit(x_data, y_data).transform(x_data).compute()
    except MemoryError:
        return None
    return time.time() - start


def resize(data, n_sample, n_chunk, n_chunk_feature):
    return data[:n_sample].rechunk(
        (n_sample // n_chunk, data.shape[1] // n_chunk_feature)
    ).persist()


PCAS = dict(
    pca_full=PCA(n_components=3, svd_solver='full'),
    pca_random=PCA(n_components=3, svd_solver='randomized'),
    pca_sk=PCA_sklearn(n_components=3),
    ipca=IncrementalPCA(n_components=3),
)

@click.command()
@click.option('--samples', default=8900, help='number of samples')
@click.option('--chunks', default=1, help='number of chunks on sample axis')
@click.option('--pca', type=click.Choice(list(PCAS.keys())), default='pca_random', help='pca choice')
@click.option('--filename', default='output.csv', help='CSV output file name')
@click.option('--n_workers', default=1, help='number of workers')
@click.option('--threads_per_worker', default=1, help='number of threads per worker')
def run(samples, chunks, pca, filename, n_workers, threads_per_worker):
    Client(
        processes=True,
        threads_per_worker=threads_per_worker,
        n_workers=n_workers,
        memory_limit="80GB"
    )

    x_data, y_data = prepare_data(samples, chunks)

    pca_time = pca_func(
        x_data,
        y_data,
        PCAS[pca],
    )

    with open(filename, 'a') as f:
        if pca_time is not None:
            pca_string = f'{pca_time:.3f}'
        else:
            pca_string = f'{pca_time}'

        f.write(f'{pca},{chunks},{samples},' + pca_string + '\n')
        print(f'{pca},{chunks},{samples},' + pca_string)

if __name__ == '__main__':
    run()
