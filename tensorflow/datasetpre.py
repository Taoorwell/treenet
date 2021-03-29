import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import tensorflow as tf

# data path
image_path = u'../../tiles/'
mask_path = u'../../masks/'

images = [image_path + i for i in os.listdir(image_path)]
masks = [mask_path + i for i in os.listdir(mask_path)]


def get_raster(raster_path):
    ds = gdal.Open(raster_path)
    arr = np.empty((ds.RasterYSize, ds.RasterXSize, ds.RasterCount))
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b).ReadAsArray()
        arr[:, :, b-1] = band
    return arr


def norma_data(data, norma_methods="z-score"):
    arr = np.empty(data.shape)
    for i in range(data.shape[-1]):
        array = data.transpose(2, 0, 1)[i, :, :]
        mins, maxs, mean, std= np.percentile(array, 1), np.percentile(array, 99), np.mean(array), np.std(array)
        if norma_methods == "z-score":
            new_array = (array-mean)/std
        else:
            new_array = np.clip(2*(array-mins)/(maxs-mins), 0, 1)
        arr[:, :, i] = new_array
    return arr


def get_patches(image_path, height, width, num):
    image = get_raster(image_path)
    if image.shape[-1] > 1:
        image = norma_data(image, norma_methods='min-max')


def random_sample(m):
    x = np.random.randint(m/2 - 1, 999 - m/2)
    y = np.random.randint(m/2 - 1, 999 - m/2)
    location = (x, y)
    return location


class Treedatasets(object):
    def __init__(self, image_path, mask_path, patch_size, number):
        self.image_path = image_path
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.number = number

    def __len__(self):
        return len(self.image_path) * self.number

    def __getitem__(self, item):
        image_id = item // self.number
        location = random_sample(self.patch_size)
        sample = {'image': self.image_path[image_id], 'mask': self.mask_path[image_id],
                  'patch': location}
        return sample


# Treedatasets = Treedatasets(image_path=images, mask_path=masks, patch_size=256, number=250)



# images = [get_raster(x) for x in images]
# print(images)








# def pare_function(image, mask):
#     image_str = tf.compat.as_str_any(image.numpy())
#     mask_str = tf.compat.as_str_any(mask.numpy())
#
#     image = get_raster(image_str)
#     image = norma_data(image, norma_methods='min-max')
#     image = tf.image.convert_image_dtype(image, tf.float32)
#
#     mask = get_raster(mask_str)
#     mask = tf.image.convert_image_dtype(mask, tf.float32)
#
#     return image, mask


# dataset = tf.data.Dataset.from_tensor_slices((images, masks))
# dataset = dataset.shuffle(len(images))
# dataset = dataset.map(lambda x, y: tf.py_function(func=pare_function, inp=[x, y], Tout=tf.string))
# dataset = dataset.map(pare_function, num_parallel_calls=4)


# for i in dataset:
#     image, mask = pare_function(i[0], i[1])
#     print(image.shape, mask.shape)
# '    image_tensor = i[0]
#     print(image_tensor)
#     image_str = tf.compat.as_str_any(image_tensor.numpy())
#     print(image_str)'

    # print(image_str)

# for image, mask in zip(images, masks):
#     image_arr = get_raster(image_path + image)
#     image_arr = norma_data(image_arr, norma_methods='min-max')
#     mask_arr = get_raster(mask_path + mask)
#     ax1 = plt.subplot2grid((1, 2), (0, 0))
#     ax1.imshow(image_arr[:, :, 1:4])
#     ax2 = plt.subplot2grid((1, 2), (0, 1))
#     ax2.imshow(mask_arr.reshape((1000, 1000)))
#     ax2.set_title(mask.split('_')[-1].split('.')[0])
#     plt.show()
