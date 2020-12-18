#!/usr/bin/env python
# coding: utf-8
# # This .py file finish the purpose on reading vector or .mat data format
# which labeling by hand for extracting train raster pixels
# # and pixels labeling accordingly, is for preparation before feeding data into models.
# # And then, load needed classification files and get classification results!


# # Load packages
import os
import time
import numpy as np
import pandas as pd
import seaborn as sn
from tqdm import tqdm
import scipy.io as sio
from osgeo import gdal
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, accuracy_score
# from sklearn.decomposition import PCA
# Functions of Gdal
# get raster data info, included rows, cols, n_bands, bands_data(read by band and shape is (W,H,C)),
# projection and geo transformation.


# # Read raster data.
def get_raster_info(raster_data_path):
    raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
    # geo_transform = raster_dataset.GetGeoTransform()
    # projection = raster_dataset.GetProjectionRef()
    bands_data = []
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())
    bands_data = np.dstack(bands_data)
    bands_data = bands_data[:, :, :]
    # rows, cols, n_bands = bands_data.shape
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


# Read and Write mat data.
def get_mat(mat_data_path):
    bands_data_dict = sio.loadmat(mat_data_path)
    bands_data = bands_data_dict[list(bands_data_dict.keys())[-1]]
    return bands_data


def get_mat_info(mat_data_path, train_mat_data_path):
    bands_data = get_mat(mat_data_path)
    labeled_pixel = get_mat(train_mat_data_path)
    is_train = np.nonzero(labeled_pixel)
    training_labels = labeled_pixel[is_train]
    return bands_data, is_train, training_labels


def save_array_to_mat(array, filename):
    dicts = {"pre": array}
    sio.savemat(filename, dicts)


# Prepare data for raster data, train_index, and train_labels
def get_prep_data(data_path, train_data_path, norma_method="z-score"):
    if data_path.endswith('.dat'):
        rows, cols, n_bands, bands_data, geo_transform, proj = get_raster_info(data_path)
        try:
            labeled_pixels, is_train = vectors_to_raster(train_data_path, data_path)
            training_labels = labeled_pixels[is_train]
        except NotADirectoryError:
            rows, cols, n_bands, band_data, geo_transform, proj = get_raster_info(train_data_path)
            band_data = band_data.reshape(rows, cols)
            is_train = np.nonzero(band_data)
            training_labels = band_data[is_train]
    else:
        bands_data, is_train, training_labels = get_mat_info(data_path, train_data_path)
    bands_data = norma_data(bands_data, norma_methods=norma_method)
    return bands_data, is_train, training_labels


def get_shuffle(a, b):

    s = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(s)
    np.random.shuffle(b)
    return a, b


# Custom train index and text index from one shapefiles.
def custom_train_index(seed, is_train, training_labels, c, lists):
    np.random.seed(seed)
    index = np.array(is_train).transpose((1, 0))
    x_train_index, x_test_index, y_train, y_test = [], [], [], []
    for i, n in zip(range(1, c+1, 1), lists):
        i_index = [j for j, x in enumerate(training_labels) if x == i]
        i_index_random = np.random.choice(i_index, n, replace=False)
        i_index_rest = [k for k in i_index if k not in i_index_random]
        i_train_index = index[i_index_random]
        i_train_labels = np.ones(len(i_index_random)) * i
        i_test_index = index[i_index_rest]
        i_test_labels = np.ones(len(i_test_index)) * i
        x_train_index.append(i_train_index)
        x_test_index.append(i_test_index)
        y_train.append(i_train_labels)
        y_test.append(i_test_labels)

    x_train_index = np.concatenate(x_train_index, axis=0)
    x_test_index = np.concatenate(x_test_index, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    x_train_index, y_train = get_shuffle(x_train_index, y_train)
    x_test_index, y_test = get_shuffle(x_test_index, y_test)
    return x_train_index, x_test_index, y_train, y_test


# Prepare train raster data and train labels for training model.
# Return train_samples and train_labels
def get_train_sample(data_path, train_data_path, c, norma_methods='z-score', m=1):
    bands_data, is_train, training_labels = get_prep_data(data_path, train_data_path,
                                                          norma_method=norma_methods)
    x_train_index = np.array(is_train).transpose((1, 0))
    samples = []
    if m == 1:
        for i in x_train_index:
            sample = bands_data[i[0], i[1]]
            samples.append(sample)
        train_samples = np.stack(samples)
        # if d == 3:
        #     train_samples = train_samples.reshape((train_samples.shape[0], train_samples.shape[1], -1))

    else:
        n = int((m - 1) / 2)
        # x_train_nindex = x_train_index + n
        # bands_data = np.pad(bands_data, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
        for j in x_train_index:
            k1 = j[0] - n
            k2 = j[0] + n + 1
            k3 = j[1] - n
            k4 = j[1] + n + 1
            block = bands_data[k1:k2, k3:k4]
            samples.append(block)
        train_samples = np.stack(samples, axis=0)
        # if d == 5:
        #     train_samples = train_samples.reshape((train_samples.shape[0], train_samples.shape[1],
        #                                            train_samples.shape[2], train_samples.shape[3], -1))
    train_labels = one_hot_encode(c, training_labels)

    return train_samples, train_labels


# According to given test shape or test index to obtain OA and Kappa from the model
# Only for model MLP and CNN
def get_test_predict(model, data_path, test_data_path, bsize, norma_methods='z-score', m=1):
    bands_data, is_test, test_labels = get_prep_data(data_path, test_data_path,
                                                     norma_method=norma_methods)
    x_test_index = np.array(is_test).transpose((1, 0))
    samples = []
    predicts = []
    if m == 1:
        for i in x_test_index:
            sample = bands_data[i[0], i[1]]
            samples.append(sample)
        samples = np.stack(samples)
        if len(model.input.shape) == 3:
            samples = samples.reshape((samples.shape[0], samples.shape[1], -1))
        predicts = model.predict(samples)

    else:
        n = int((m - 1) / 2)
        # x_test_nindex = x_test_index + n
        # bands_data = np.pad(bands_data, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
        for i, j in enumerate(x_test_index):
            k1 = j[0] - n
            k2 = j[0] + n + 1
            k3 = j[1] - n
            k4 = j[1] + n + 1
            block = bands_data[k1:k2, k3:k4]
            samples.append(block)
            if len(samples) == bsize or i == x_test_index.shape[0] - 1:
                print("Batches Predictions...")
                pre = np.stack(samples)
                if len(model.input.shape) == 5:
                    pre = pre.reshape((pre.shape[0], pre.shape[1], pre.shape[2], pre.shape[3], -1))
                predict = model.predict(pre)
                predicts.append(predict)
                samples = []
        predicts = np.concatenate(predicts)
    print("Batches Predictions Finish!!!")
    oa, kappa = print_plot_cm(test_labels, predicts)
    return oa, kappa


# Write out whole research region predicts
# Return array, whose shape is (R, C, N_CLASS)
def write_region_predicts(model, image_data_path, region_data_path,
                          bsize, filename, norma_methods='z-score', m=1):
    print("Step1: Start Predicting Whole Region...")
    bands_data, is_train, _ = get_mat_info(image_data_path, region_data_path)
    bands_data = norma_data(bands_data, norma_methods)
    index = np.array(is_train).transpose((1, 0))
    samples = []
    if m == 1:
        for i in index:
            sample = bands_data[i[0], i[1]]
            samples.append(sample)
        samples = np.stack(samples)
        if len(model.input.shape) == 3:
            samples = samples.reshape((samples.shape[0], samples.shape[1], -1))
        predicts = model.predict(samples)
    else:
        predicts = []
        n = int((m - 1) / 2)
        t1 = time.clock()
        for i, j in tqdm(enumerate(index)):
            k1 = j[0] - n
            k2 = j[0] + n + 1
            k3 = j[1] - n
            k4 = j[1] + n + 1
            block = bands_data[k1:k2, k3:k4]
            samples.append(block)
            if len(samples) == bsize or i == index.shape[0] - 1:
                # print("Batches Predictions...")
                pre = np.stack(samples)
                if len(model.input.shape) == 5:
                    pre = pre.reshape((pre.shape[0], pre.shape[1], pre.shape[2], pre.shape[3], -1))
                pre = model.predict(pre)
                predicts.append(pre)
                del samples
                samples = []
        predicts = np.concatenate(predicts)
        t2 = time.clock()
        print("    Batches Predictions Finish!!!")
        print("    Time Consuming: {}".format(t2 - t1))
    print("Step2: Begin Calculator Predicts and Probabilities...")
    shape = (7500, 5000, 2)
    labeled_pixel_dict = sio.loadmat(region_data_path)
    labeled_pixel = labeled_pixel_dict[list(labeled_pixel_dict.keys())[-1]]
    is_train = np.nonzero(labeled_pixel)
    labels = np.argmax(predicts, axis=-1) + 1
    probs = np.max(predicts, axis=1)
    result = np.zeros(shape)
    for i, j, k, p in zip(is_train[0], is_train[1], labels, probs):
        result[i, j, 0] = k
        result[i, j, 1] = p
    print("    Saving Predicts and Probabilities into Mat File....")
    save_array_to_mat(result, filename=filename)
    print("    Save Predicts Success Check in " + filename)
    # print("Step3: Start Plotting Classification and Confidence map....")
    # plot_region_image_classification_result_prob(filename)
    print("ALL TASK FINISH!!!")


# Write out predicts which is a array to mat file.
# mat_file is an array whose shape is (R, C, 2), one is predicts, other one is max prob.
# plot predicts and prob
def write_region_image_classification_result_probs(predict, train_data_path, shape, filename):
    labeled_pixel_dict = sio.loadmat(train_data_path)
    labeled_pixel = labeled_pixel_dict[list(labeled_pixel_dict.keys())[-1]]
    is_train = np.nonzero(labeled_pixel)
    # if predict.ndim == 2:
    #     labels = np.argmax(predict, axis=-1) + 1
    # else:
    #     labels = predict
    labels = np.argmax(predict, axis=-1) + 1
    probs = np.max(predict, axis=1)
    result = np.zeros(shape)
    for i, j, k, p in zip(is_train[0], is_train[1], labels, probs):
        result[i, j, 0] = k
        result[i, j, 1] = p
    # print("Plotting the Results and Probabilities...")
    # plt.subplot(121)
    # plot_predicts(result[:, :, 0])
    # plt.subplot(122)
    # sn.heatmap(result[:, :, 1], annot=False, cmap="Greys_r", xticklabels=False, yticklabels=False)
    # plt.savefig(filename.split(".")[0] + ".png")
    print("Saving Predicts and Probabilities into Mat File....")
    save_array_to_mat(result, filename=filename)
    print("Save Predicts Success Check in " + filename)


def plot_region_image_classification_result_prob(predict_mat_path):
    result = get_mat(predict_mat_path)
    plt.subplot(121)
    plot_predicts(result[:, :, 0])
    # plt.xlabel("Classification Predict Map")
    plt.subplot(122)
    prob = result[:, :, 1]
    prob[prob == 0] = np.nan
    plt.imshow(prob, cmap='YlOrRd_r')
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel("Classification Confidence Map")
    plt.colorbar()
    # plt.axis('off')
    # plt.show()


# Using CNN model to predict each segments and obtain accordingly predict value and probabilities.
# Return segments shapefiles and add two field value, predicts and prob.
# In order to save into shape file,segment.to_file(file_path)
def get_predicts_segments(segments_path, raster_data_path, test_data_path, image_mat_path, norma_methods, m, model,
                          filename):
    segments = gpd.read_file(segments_path)
    print("Step1: Begin Generating Centroid and Predicting...")
    t1 = time.clock()
    x = segments.centroid.x
    y = segments.centroid.y
    segments['R'] = [int(a) for a in (2957730.452 - y)]
    segments["C"] = [int(b) for b in (x - 541546.573)]
    l = len(segments['R'])
    bands_data = get_mat(mat_data_path=image_mat_path)
    bands_data = norma_data(bands_data, norma_methods)
    n = int((m - 1) / 2)
    samples = []
    pres = []
    q = l//20000
    # y = l % 20000
    p = 0
    for x, y in tqdm(zip(segments['R'], segments['C'])):
        k1 = x - n
        k2 = x + n + 1
        k3 = y - n
        k4 = y + n + 1
        block = bands_data[k1:k2, k3:k4]
        samples.append(block)
        if len(samples) == 20000 or (len(samples) + p*20000 == l):
            print("    Starting Predicts Segments")
            pre = model.predict(np.stack(samples))
            pres.append(pre)
            samples = []
            p = p + 1
    press = np.concatenate(pres)
    t2 = time.clock()
    T = t2 - t1
    print("    Predicting time {}".format(T))
    # print("    Predicting Finish!!!")
    predicts = np.argmax(press, axis=-1) + 1
    number = len(predicts)
    print("Number of Segments: {}".format(number))
    prob = np.max(press, axis=1)

    print("Step2: Start Saving Predicts and Probabilities...")
    segments['predicts'] = predicts
    segments['prob'] = prob
    print("    Save Results into Shapefiles Success!!")
    segments.to_file(filename)
    print("    Check Shapefile in {}".format(filename))

    print("Step3: Begin Rasterilizing Shapefiles into Raster and Test...")
    predicts, index = vectors_to_raster(vector_data_path=filename, raster_data_path=raster_data_path, field="predicts")
    oa, kappa = get_test_segments(data_path=image_mat_path, test_data_path=test_data_path, predicts=predicts)
    del predicts, pres, press, samples, segments
    # prob, index = vectors_to_raster(vector_data_path=filename, raster_data_path=raster_data_path, field='prob')

    print("Step4: Start Plotting Classification and Confidence Map")
    ax1 = plt.subplot(121)
    plot_predicts(predicts)
    ax1.set_xlabel("Classification Predict Map")

    ax2 = plt.subplot(122)
    segments.plot(column='prob', cmap='YlOrRd_r', ax=ax2, legend=True)
    # plt.imshow(prob, cmap='Greys')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("Classification Confidence Map")
    plt.colorbar()
    plt.axis('off')
    plt.show()
    print("ALL TASK FINISH!!!")
    return oa, kappa


def plot_segments_predicts_prob(segment_path, raster_data_path, field='predicts'):
    predicts, index = vectors_to_raster(vector_data_path=segment_path, raster_data_path=raster_data_path,
                                        field=field)
    segments = gpd.read_file(segment_path)
    ax1 = plt.subplot(121)
    plot_predicts(predicts)
    # ax1.set_xlabel("Classification Predict Map")

    ax2 = plt.subplot(122)
    segments.plot(column='prob', cmap='YlOrRd_r', ax=ax2, legend=True)
    # plt.imshow(prob, cmap='Greys')
    ax2.set_xticks([])
    ax2.set_yticks([])
    # ax2.set_xlabel("Classification Confidence Map")
    # plt.colorbar()
    # plt.axis('off')
    # plt.show()


# Get test accuracy from the segment predict
# predicts is raster, after get segment, output as shapefiles, and vector_to_raster.
def get_test_segments(data_path, test_data_path, predicts, norma_methods="z-score"):
    bands_data, is_test, test_labels = get_prep_data(data_path, test_data_path,
                                                     norma_method=norma_methods)
    predicts_label = predicts[is_test]
    oa, kappa = print_plot_cm(test_labels, predicts_label)
    return oa, kappa


# Some additional functions.
def plot_predicts(arr_2d):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    plt.imshow(arr_3d)
    plt.xticks([])
    plt.yticks([])
    C1 = mpatches.Patch(color='Magenta', label='KYL')
    C2 = mpatches.Patch(color='Lime', label='ZLD')
    C3 = mpatches.Patch(color='ForestGreen', label='SML')
    C4 = mpatches.Patch(color='Red', label='MWS')
    C5 = mpatches.Patch(color='Coral', label='CFJD')
    C6 = mpatches.Patch(color='DeepSkyBlue', label='SLD')
    C7 = mpatches.Patch(color='Cyan', label='WCLD')
    plt.legend(handles=[C1, C2, C3, C4, C5, C6, C7])
    # plt.axis('off')
    # plt.show()


def print_plot_cm(y_true, y_pred):
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=-1) + 1
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=-1) + 1
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    print("Overall Accuracy:{:.4%}".format(oa))
    print("Kappa: ", kappa)
    print(classification_report(y_true, y_pred, digits=4))
    labels = sorted(list(set(y_true)))
    cm_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm_data, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='0000')
    plt.show()
    return oa, kappa


# # statics samples classes info form labels. return a dict.
def get_samples_info(labels_samples):
    unique, counts = np.unique(labels_samples, return_counts=True)
    return dict(zip(unique, counts))


# # one-hot encoding for labels, return shape of (c,1), value is form  1 to c
# # parameter: c, number of classes, labels, train valid test label data, form is [1, 2, 1, 0, .....]
def one_hot_encode(c, labels):
    return np.eye(c)[[int(e) for e in labels]]


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


# As for labeling, one pixel might be labeled more twice.So we delete those pixel by index.
# def delete_error_category(training_labels, training_samples):
#     category = np.unique(training_labels)
#     for i in category[20:]:
#         index = np.argwhere(training_labels == i)
#         training_labels = np.delete(training_labels, index)
#         training_samples = np.delete(training_samples, index, axis=0)
#     return training_samples, training_labels


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


# # read train_data form .mat for converting to rgb color.
def convert_to_color(train_data_path):
    train_mat_dict = sio.loadmat(train_data_path)
    arr_2d = train_mat_dict[list(train_mat_dict.keys())[-1]]
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    plt.imshow(arr_3d)
    plt.show()


# # plot model training history, context of history included train and valid loss and accuracy.
# # parameter network, network == model.fit().
def plot_history(network):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(network.history["loss"])
    plt.plot(network.history["val_loss"])
    plt.legend(["Training", "Validation"])

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(network.history["acc"])
    plt.plot(network.history["val_acc"])
    plt.legend(["Training", "Validation"])
    plt.show()

# # increase plotting result function more.....


# # Over-sampling ways
# # random oversampling ways
# def ros(x_train, y_train):
#     over = RandomOverSampler()
#     x_train_over, y_train_over = over.fit_sample(x_train, y_train)
#     return x_train_over, y_train_over
#
#
# # SMOTE Ways
# def smote(x_train, y_train):
#     over1 = SMOTE()
#     x_train_over1, y_train_over1 = over1.fit_sample(x_train, y_train)
#     return x_train_over1, y_train_over1


# def txt2xls(txt_path, xls_path, column):
#     df = pd.read_csv(txt_path, sep='\t', header=None)
#     new_df = df.iloc[np.arange(2, len(df), 3)]
#     new_df = new_df[0].str.split(',', expand=True)
#     l = []
#     for i in range(0, len(column)):
#         nd = new_df[i]
#         nd = nd.str.split(':', expand=True)
#         nd = nd.drop(0, axis=1)
#         nd.rename(columns={1: column[i]}, inplace=True)
#         l.append(nd)
#     ne_d = pd.concat(l, axis=1)
#     ne_d[column[1:]] = ne_d[column[1:]].apply(pd.to_numeric)
#     ne_d.to_excel(xls_path, index=False)
def split_vector(vector_path, save_path):
    samples = gpd.read_file(vector_path)
    classes = np.unique(samples['CLASS_ID'])
    for i in classes:
        single = samples[samples['CLASS_ID'] == i]
        single.to_file(save_path + r'{}.shp'.format(i))
    print("Split Finish" + " Check in " + save_path)


def split_segments_predicts(segments, save_path):
    classes = np.unique(segments['predicts'])
    for i in classes:
        single = segments[segments['predicts'] == i]
        single.to_file(save_path + '{}.shp'.format(i))
    print('Split Finish ' + " Check in " + save_path)


def one_hot(data):
    arr_3d = np.zeros((data.shape[0], data.shape[1], 2), dtype=np.uint8)
    dic = {0: (1, 0),
           1: (0, 1)}
    for k, i in dic.items():
        g = data == k
        arr_3d[g] = i
    return arr_3d



# def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
#                             projection, target_value=1):
#     data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
#     layer = data_source.GetLayer(0)
#     driver = gdal.GetDriverByName('MEM')
#     target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
#     target_ds.SetGeoTransform(geo_transform)
#     target_ds.SetProjection(projection)
#     gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
#     return target_ds
#
#
# def vectors_to_raster(vector_path, rows, cols, geo_transform, projection):
#     """Rasterize the vectors in given directory in a single image."""
#     files = [f for f in os.listdir(vector_path) if f.endswith('.shp')]
#     classes = [f.split('.')[0] for f in files]
#     shapefiles = [os.path.join(vector_path, f) for f in files]
#     labeled_pixels = np.zeros((rows, cols))
#     for i, path in zip(classes, shapefiles):
#         label = int(i)
#         ds = create_mask_from_vector(path, cols, rows, geo_transform,
#                                      projection, target_value=label)
#         band = ds.GetRasterBand(1)
#         labeled_pixels += band.ReadAsArray()
#     is_train = np.nonzero(labeled_pixels)
#     return labeled_pixels, is_train


# def vectors_to_raster1(vector_path, rows, cols, geo_transform, projection):
#     labeled_pixels = np.zeros((rows, cols))
#     ds = create_mask_from_vector(vector_path, cols, rows, geo_transform,
#                                  projection, target_value=1)
#     band = ds.GetRasterBand(1)
#     labeled_pixels += band.ReadAsArray()
#     return labeled_pixels
# # due to hyperspectral images datasets on web is .mat format, using scipy.sio read .mat data
# # return is dict and bands data and labels ndarray is last key value.
# # get is_train from labels ndarry and generate training labels and bands data and is_train.
# def write_out_whole_predicts(model, data_path, bsize, norma_methods='z-score', m=1):
#     # bands_data_dict = sio.loadmat(data_path)
#     # bands_data = bands_data_dict[list(bands_data_dict.keys())[-1]]
#     rows, cols, n_bands, bands_data, geo_transform, proj = get_raster_info(
#         raster_data_path=data_path)
#     bands_data = norma_data(bands_data, norma_methods=norma_methods)
#     # if pca is True:
#     #     bands_data = pca_data(bands_data, n=n)
#     if m == 1:
#         if len(model.input.shape) == 2:
#             pre = bands_data.reshape((bands_data.shape[0]*bands_data.shape[1], bands_data.shape[2]))
#         else:
#             pre = bands_data.reshape((bands_data.shape[0]*bands_data.shape[1], bands_data.shape[2], -1))
#         predicts = model.predict(pre)
#     else:
#         n = int((m - 1) / 2)
#         bands_data = np.pad(bands_data, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
#         cols = bands_data.shape[1]-2*n
#         rows = bands_data.shape[0]-2*n
#         result = []
#         predicts = []
#         for i in range(0, rows, 1):
#             for j in range(0, cols, 1):
#                 data = bands_data[i: i + m, j: j + m, :]
#                 result.append(data)
#                 if len(result) == bsize or i == int(rows-1):
#                     print("Batches Predictions...")
#                     pre = np.stack(result)
#                     if len(model.input.shape) == 5:
#                         pre = pre.reshape((pre.shape[0], pre.shape[1], pre.shape[2], pre.shape[3], -1))
#                     predict = model.predict(pre)
#                     predicts.append(predict)
#                     result = []
#         predicts = np.concatenate(predicts)
#         print("Batches Predictions Finish!!!")
#     return predicts
#     # write_classification_result2(predicts, shape)
#     # write_classification_prob(predicts, shape)
# def get_fusion_features_from_test(model1, model2, data_path, train_data_path, c, lists, m):
#     extractor_from_model1, extractor_from_model2 = feature_extractor(model1, model2)
#     bands_data, is_train, train_labels = get_prep_data(data_path, train_data_path)
#     _, test_index, _, y_test = custom_train_index(is_train, train_labels, c, lists)
#     features1 = []
#     for i in test_index:
#         feature = bands_data[i[0], i[1]]
#         features1.append(feature)
#     features1 = np.stack(features1)
#     features1 = features1.reshape((features1.shape[0], features1.shape[1], -1))
#     features1 = extractor_from_model1([features1])[0]
#     features2 = []
#     samples = []
#     n = int((m - 1) / 2)
#     x_test_nindex = test_index + n
#     bands_data = np.pad(bands_data, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
#     for i, j in enumerate(x_test_nindex):
#         k1 = j[0] - n
#         k2 = j[0] + n + 1
#         k3 = j[1] - n
#         k4 = j[1] + n + 1
#         block = bands_data[k1:k2, k3:k4]
#         samples.append(block)
#         if len(samples) == 3200 or i == x_test_nindex.shape[0] - 1:
#             print("Batches Features...")
#             pre = np.stack(samples)
#             feature = extractor_from_model2([pre])[0]
#             features2.append(feature)
#             samples = []
#     features2 = np.concatenate(features2)
#     fusion_features = np.concatenate([features1, features2], axis=1)
#     return fusion_features, y_test


# def get_fusion_features_from_whole(model1, model2, data_path, m):
#     extractor_from_model1, extractor_from_model2 = feature_extractor(model1, model2)
#     bands_data_dict = sio.loadmat(data_path)
#     bands_data_1 = bands_data_dict[list(bands_data_dict.keys())[-1]]
#     bands_data_1 = norma_data(bands_data_1)
#     features1 = bands_data_1.reshape((bands_data_1.shape[0] * bands_data_1.shape[1], bands_data_1.shape[2], -1))
#     f1 = extractor_from_model1([features1])[0]
#
#     n = int((m - 1) / 2)
#     bands_data_1 = np.pad(bands_data_1, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
#     cols = bands_data_1.shape[1] - 2 * n
#     rows = bands_data_1.shape[0] - 2 * n
#     result1 = []
#     f2 = []
#     for g in range(0, rows, 1):
#         for h in range(0, cols, 1):
#             data = bands_data_1[g: g + m, h: h + m, :]
#             result1.append(data)
#             if len(result1) == 1600 or g == int(rows - 1):
#                 print("Batches Features...")
#                 pre1 = np.stack(result1)
#                 fe = extractor_from_model2([pre1])[0]
#                 f2.append(fe)
#                 result1 = []
#     f2 = np.concatenate(f2)
#
#     f3 = np.concatenate([f1, f2], axis=1)
#     return f3


# def write_classification_result_tif(fname, classification, original_raster_data_path, m=1):
#     """Create a GeoTIFF file with the given data."""
#     rows, cols, n_bands, bands_data, geo_transform, proj = get_raster_info(raster_data_path=original_raster_data_path)
#     driver = gdal.GetDriverByName('GTiff')
#     classification = classification.reshape((rows, cols))
#     dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte)
#     dataset.SetGeoTransform(geo_transform)
#     dataset.SetProjection(proj)
#     band = dataset.GetRasterBand(1)
#     band.WriteArray(classification)
#     dataset = None  # close the file


# # write out whole image to RGB
# # predict value form model, original image shape
# def write_whole_image_classification_result(predict, shape):
#     if predict.ndim == 2:
#         predict = np.argmax(predict, axis=-1) + 1
#     arr_2d = np.reshape(predict, shape)
#     plot_predicts(arr_2d)


# def write_whole_image_predicts_prob(predict, shape):
#     max_prob = np.max(predict, axis=1)
#     mean_prob = np.mean(predict, axis=1)
#     sec_max = []
#     for p in predict:
#         p = sorted(p)
#         second = p[-2]
#         sec_max.append(second)
#     conf1 = max_prob - sec_max
#     conf2 = max_prob - mean_prob
#     low_confidence0 = [x for x in max_prob if x < 0.5]
#     low_confidence1 = [x for x in conf1 if x < 0.5]
#     low_confidence2 = [x for x in conf2 if x < 0.5]
#     cof1 = len(low_confidence0)/(shape[0]*shape[1])
#     cof2 = len(low_confidence1)/(shape[0]*shape[1])
#     cof3 = len(low_confidence2)/(shape[0]*shape[1])
#     # print("cof1:{}, cof2:{}, cof3: {}".format(cof1, cof2, cof3))
#     # prob_img = np.reshape(conf, shape)
#     # fig = plt.figure()
#     # fig.add_subplot(121)
#     # plt.xlabel("Confidences")
#     # plt.ylabel("Numbers")
#     # plt.hist(conf, bins=10, range=(0, 1), facecolor='red', alpha=0.5)
#     #
#     # fig.add_subplot(122)
#     # sn.heatmap(prob_img, annot=False, cmap="Greys_r", xticklabels=False, yticklabels=False)
#     # plt.imshow(prob_img, cmap='gray')
#     #
#     # plt.show()
#     return cof1, cof2, cof3

