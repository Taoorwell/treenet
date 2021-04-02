import tensorflow as tf
from utility import *
from residual_unet import *
from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

width = 128
batch_size = 5
# Datasets construction


# def dataset_generator(path, mode, width, batch_size):
#     images_path, masks_path = load_data(path, mode)
#     datasets = tf.data.Dataset.from_tensor_slices((images_path, masks_path))
#     ds = datasets.map(pare_fun)
#     while True:
#         for images, masks in ds:
#             image_patch = np.zeros((batch_size, width, width, 7))
#             mask_patch = np.zeros((batch_size, width, width, 1))
#             for i in range(batch_size):
#                 location = np.random.randint(width / 2 - 1, 999 - width / 2, (2,))
#                 h, w = location[0], location[1]
#                 d1 = int(width / 2 - 1)
#                 d2 = int(width - d1)
#                 image_patch[i] = images[h - d1: h + d2, w - d1: w + d2, :]
#                 mask_patch[i] = masks[h - d1: h + d2, w - d1: w + d2]
#             yield image_patch, mask_patch

def image_dataset(path, mode, width, batch_size):
    # image path and mask path dataset
    images_path, masks_path = load_data(path, mode)
    datasets = tf.data.Dataset.from_tensor_slices((images_path, masks_path))
    # datasets = datasets.repeat()

    # parse path into full image and then into patches
    # define parse function
    def parse_fun(x, y):
        def f(x, y):
            x1 = x.decode()
            y1 = y.decode()

            x2 = get_raster(x1)
            y2 = get_raster(y1)

            image_patch = np.zeros((batch_size, width, width, 7), dtype=np.float32)
            mask_patch = np.zeros((batch_size, width, width, 1), dtype=np.float32)

            for i in range(batch_size):
                location = np.random.randint(width // 2 - 1, 999 - width // 2, (2,))
                h, w = location[0], location[1]
                d1 = int(width / 2 - 1)
                d2 = int(width - d1)
                image_patch[i] = x2[h - d1: h + d2, w - d1: w + d2, :]
                mask_patch[i] = y2[h - d1: h + d2, w - d1: w + d2]
            return image_patch, mask_patch

        image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
        image.set_shape([batch_size, width, width, 7])
        mask.set_shape([batch_size, width, width, 1])
        return image, mask
    datasets = datasets.map(parse_fun)
    return datasets

# for image, mask i, n ds:
#     print(image.shape, mask.shape)
#     plt.subplot(121)
#     plt.imshow(image.numpy()[0][:, :, 0:3])
#     plt.subplot(1, 2, 2)
#     plt.imshow(mask.numpy()[0][:, :, 0])
#     plt.show()


train_dataset = image_dataset(path='../../', mode='train',
                              width=width, batch_size=batch_size)
#
val_dataset = image_dataset(path='../..', mode='eval',
                            width=width, batch_size=batch_size)

# model construction
model = build_res_unet(input_shape=(width, width, 7))
# model.summary()

# model compile
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
              loss=dice_loss, metrics=[dice])

# tensorboard
tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir='tb_callback_dir',
                                                       histogram_freq=1)
# model train and validation
# train_writer = tf.summary.create_file_writer("logs/train/")
# test_writer = tf.summary.create_file_writer("logs/test/")
# train_step = test_step = 0
#
#
# for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
#     train_step = test_step = 0
#     train_writer = tf.summary.create_file_writer("logs/train/" + str(lr))
#     test_writer = tf.summary.create_file_writer("logs/test/" + str(lr))
#     model = build_res_unet(input_shape=(width, width, 7))
#     optimizer = tf.optimizers.Adam(learning_rate=lr)
#
#     for epoch in range(num_epochs):
#         # Iterate through training set
#         for batch_idx, (x, y) in enumerate(train_dataset):
#             with tf.GradientTape() as tape:
#                 y_pred = model(x, training=True)
#                 loss = dice_loss(y, y_pred)
#
#             gradients = tape.gradient(loss, model.trainable_weights)
#             optimizer.apply_gradients(zip(gradients, model.trainable_weights))
#
#             with train_writer.as_default():
#                 tf.summary.scalar("Loss", loss, step=train_step)
#                 tf.summary.scalar(
#                     "Accuracy", dice(y, y_pred), step=train_step,
#                 )
#                 train_step += 1
#
#         # Reset accuracy in between epochs (and for testing and test)
#
#         # Iterate through test set
#         for batch_idx, (x, y) in enumerate(val_dataset):
#             y_pred = model(x, training=False)
#             loss = dice_loss(y, y_pred)
#
#             with test_writer.as_default():
#                 tf.summary.scalar("Loss", loss, step=test_step)
#                 tf.summary.scalar(
#                     "Accuracy", dice(y, y_pred), step=test_step,
#                 )
#                 test_step += 1
model.fit(train_dataset, steps_per_epoch=100, epochs=50,
          validation_data=val_dataset, validation_steps=10,
          callbacks=[tensorboard_callbacks])
# model.save('model.h5')
model.save_weights('ckpt')








