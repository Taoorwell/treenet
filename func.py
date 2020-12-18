#!/usr/bin/env python
# coding: utf-8
# Load packages
import os
import torch
import numpy as np
from osgeo import gdal
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


# # Read raster data.
def get_raster_info(raster_data_path):
    raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
    bands_data = []
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())
    bands_data = np.dstack(bands_data)
    bands_data = bands_data[:, :, :]
    return bands_data


# Read shapefiles
# Read shapefiles of label, And rasterize layer with according label values.used together with below func.
def vectors_to_raster(vector_data_path, raster_data_path, field="CLASS_ID"):
    rows, cols, n_bands, bands_data, geo_transform, projection = get_raster_info(raster_data_path)
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)

    gdal.RasterizeLayer(target_ds, [1], layer, None, None, [0], ['ALL_TOUCHED=FALSE', 'ATTRIBUTE={}'.format(field)])

    labeled_pixels = target_ds.GetRasterBand(1).ReadAsArray()
    is_train = np.nonzero(labeled_pixels)
    return labeled_pixels, is_train


# # norma bands data(whole image), calculator each bands max mean min std value to obtain norma info.
# # norma bands data according to norma info, method including z-score and max-min.
# # parameter: bands data, and norma method.
def norma_data(data, norma_methods="z-score"):
    norma_info = []
    for i in range(1, data.shape[-1]):
        array = data.transpose(2, 0, 1)[i, :, :]
        mins = np.min(array)
        maxs = np.max(array)
        mean = np.mean(array)
        std = np.std(array)
        lists = [mins, maxs, mean, std]
        norma_info.append(lists)
    norma_info = np.stack(norma_info, axis=0)
    new_data = []
    for i, j in zip(range(norma_info.shape[0]), range(1, data.shape[-1])):
        norma_info1 = norma_info[i, :]
        array = data[:, :, j]
        if norma_methods == "z-score":
            new_array = (array-norma_info1[2])/norma_info1[3]
        else:
            new_array = 2*(array-norma_info1[0])/(norma_info1[1]-norma_info1[0])
        new_data.append(new_array)
    new_data = np.stack(new_data, axis=-1)
    # # for save half memory!
    new_data = np.float32(new_data)
    return new_data


def get_image_data(file):
    bands_data = get_raster_info(file)
    image_data = norma_data(bands_data, norma_methods='min-max')
    return image_data


class CrownDataset(Dataset):
    def __init__(self, tif_file, mask_file, m, n_random):
        self.tif_file = tif_file
        self.mask_file = mask_file
        self.m = m
        self.n_random = n_random

    def __len__(self):
        return len(self.tif_file) * self.n_random

    def __getitem__(self, item):
        i = item // self.n_random
        image_data = get_image_data(self.tif_file[i])
        mask_data = get_raster_info(self.mask_file[i])
        location = random_sample(self.m)
        h, w = location[0], location[1]
        d1 = int(self.m/2 - 1)
        d2 = int(self.m - d1)
        patch = torch.from_numpy(image_data[h-d1: h+d2, w-d1: w+d2, :].transpose([2, 0, 1]))
        mask = torch.from_numpy(mask_data[h-d1: h+d2, w-d1: w+d2][:, :, 0])
        sample = {'patch': patch, 'mask': mask, 'location': (h, w)}
        return sample


def random_sample(m):
    x = np.random.randint(m/2 - 1, 999 - m/2)
    y = np.random.randint(m/2 - 1, 999 - m/2)
    location = (x, y)
    return location


def show_sample(sample):
    patch = np.array(sample['patch']).transpose((1, 2, 0))[:, :, 3:6]
    mask = np.array(sample['mask'])
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(patch)
    ax2.imshow(mask)
    plt.title('Location:{}'.format(sample['location']))
    plt.show()

# # palette is color map for rgb convert. preference setting.
# # including 16 types color, can increase or decrease.
palette = {0: (255, 255, 255),  # White
           6: (0, 191, 255),  # DeepSkyBlue
           3: (34, 139, 34),  # ForestGreen
           1: (255, 0, 255),  # Magenta
           2: (0, 255, 0),  # Lime
           5: (255, 127, 80),  # Coral
           4: (255, 0, 0),  # Red
           7: (0, 255, 255),  # Cyan
           8: (0, 255, 0),  # Lime
           9: (0, 128, 128),
           10: (128, 128, 0),
           11: (255, 128, 128),
           12: (128, 128, 255),
           13: (128, 255, 128),
           14: (255, 128, 255),
           15: (165, 42, 42),
           16: (175, 238, 238)}

