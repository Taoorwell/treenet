import os
from glob import glob
import numpy as np
from osgeo import gdal
from matplotlib import pyplot as plt


def load_data(path, mode):
    images_path = sorted(glob(os.path.join(path, "tiles/*")))
    masks_path = sorted(glob(os.path.join(path, "masks/*")))
    if mode == 'train':
        image_path, mask_path = images_path[:25], masks_path[:25]
    elif mode == 'eval':
        image_path, mask_path = images_path[25:30], masks_path[25:30]
    else:
        image_path, mask_path = images_path[30:], masks_path[30:]
    return image_path, mask_path


def get_raster(raster_path):
    ds = gdal.Open(raster_path)
    data = np.empty((ds.RasterYSize, ds.RasterXSize, ds.RasterCount), dtype=np.float32)
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b).ReadAsArray()
        data[:, :, b-1] = band
    if data.shape[-1] > 1:
        data = norma_data(data, norma_methods='min-max')
    return data


def norma_data(data, norma_methods="z-score"):
    arr = np.empty(data.shape, dtype=np.float32)
    for i in range(data.shape[-1]):
        array = data.transpose(2, 0, 1)[i, :, :]
        mins, maxs, mean, std= np.percentile(array, 1), np.percentile(array, 99), np.mean(array), np.std(array)
        if norma_methods == "z-score":
            new_array = (array-mean)/std
        else:
            new_array = np.clip(2*(array-mins)/(maxs-mins), 0, 1)
        arr[:, :, i] = new_array
    return arr




