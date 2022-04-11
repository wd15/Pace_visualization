import dask.array as da
import numpy as np
from numpy.random import default_rng
from toolz import pipe

path_ = '../data/'

x_data = da.from_zarr(path_ + "x_data.zarr" , chunks=(100, -1))
#y_data = np.load(path_ + "y_data.npy")

#index = np.random.permutation(np.arange(len(x_data)))

#x_shuffle= np.array(x_data)[index]
#y_shuffle = y_data[index]

#x_da = da.from_array(x_shuffle, chunks=(100, -1))

#da.to_zarr(path_ + 'x_data_shuffled.zarr', x_da)
#np.dump(path_ + 'x_data_shuffled.npy', y_data)
                                  
