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
import itertools
from dask.distributed import Client, LocalCluster
import click
from dask_ml.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def get_model(batch_size):
    return Pipeline([
        ("reshape", GenericTransformer(
            lambda x: x.reshape(x.shape[0], 51, 51,51)
        )),
        ("discritize",PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0)),
        ("correlations",TwoPointCorrelation(periodic_boundary=True, correlations=[(0, 0)])),
        ('flatten', GenericTransformer(lambda x: x.reshape(x.shape[0], -1))),
        ('pca', IncrementalPCA(n_components=3, svd_solver='randomized', batch_size=batch_size)),
        ('poly', PolynomialFeatures(degree=4)),
        ('regressor', LinearRegression())
    ])


def prepare_data(n_sample, n_chunk):
    x_data = da.from_zarr("/storage/home/hhive1/byucel3/data/homogenizationdata/x_data.zarr" , chunks=(100, -1))
    y_data = np.load("/storage/home/hhive1/byucel3/data/homogenizationdata/y_data.npy")
    y_data = da.from_array(y_data, chunks=(100, -1))

    # resize data
    x_data = x_data[:n_sample].rechunk((n_sample // n_chunk,) +  x_data.shape[1:])
    y_data = y_data[:n_sample].rechunk((n_sample // n_chunk,) +  y_data.shape[1:])

    return x_data.persist(), y_data.persist()

def run_fit(model, x_data, y_data):
    start = time.time()
    try:
        model.fit(x_data, y_data)
    except MemoryError:
        return None
    return model, (time.time() - start)


def run_predict(model, x_data):
    start = time.time()
    try:
        model.predict(x_data)
    except MemoryError:
        return None
    return time.time() - start



@click.command()
@click.option('--samples', default=8900, help='number of samples')
@click.option('--chunks', default=1, help='number of chunks on sample axis')
@click.option('--filename', default='output.csv', help='CSV output file name')
@click.option('--n_workers', default=1, help='number of workers')
@click.option('--threads_per_worker', default=1, help='number of threads per worker')
@click.option('--batch_size', default=1000, help='PCA batch size')
def run(samples, chunks, filename, n_workers, threads_per_worker, batch_size):
    client = Client(
        processes=True,
        threads_per_worker=threads_per_worker,
        n_workers=n_workers,
        memory_limit="160GB"
    )    

    x_data, y_data = prepare_data(samples, chunks)

    model = get_model(batch_size)
    
    model, fit_time = run_fit(model, x_data, y_data)

    predict_time = run_predict(model, x_data)

    to_str = lambda x: f'{x:.3f}' if x is not None else f'{x}'
    
    with open(filename, 'a') as f:
        f.write(f'{chunks},{samples},' + to_str(fit_time) + ',' + to_str(predict_time) + '\n')

if __name__ == '__main__':
    run()
