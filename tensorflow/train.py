from utility import *
from residual_unet import *


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


# Datasets construction
def pare_fun(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        x = get_raster(x)
        y = get_raster(y)

        return x, y

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    image.set_shape([1000, 1000, 7])
    mask.set_shape([1000, 1000, 1])
    return image, mask


def dataset_generator(path, mode, width, batch_size):
    images_path, masks_path = load_data(path, mode)
    datasets = tf.data.Dataset.from_tensor_slices((images_path, masks_path))
    ds = datasets.map(pare_fun)
    while True:
        for images, masks in ds:
            image_patch = np.zeros((batch_size, width, width, 7))
            mask_patch = np.zeros((batch_size, width, width, 1))
            for i in range(batch_size):
                location = np.random.randint(width / 2 - 1, 999 - width / 2, (2,))
                h, w = location[0], location[1]
                d1 = int(width / 2 - 1)
                d2 = int(width - d1)
                image_patch[i] = images[h - d1: h + d2, w - d1: w + d2, :]
                mask_patch[i] = masks[h - d1: h + d2, w - d1: w + d2]
            yield image_patch, mask_patch


train_dataset_generator = dataset_generator(path='../../', mode='train',
                                            width=128, batch_size=10)

val_dataset_generator = dataset_generator(path='../..', mode='eval',
                                          width=128, batch_size=10)

# model construction
model = build_res_unet(input_shape=(128, 128, 7))
# model.summary()

# model compile
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
              loss=dice_loss, metrics=[dice])

# tensorboard
# tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir='tb_callback_dir',
#                                                        histogram_freq=1)

# model train and validation
model.fit(train_dataset_generator, steps_per_epoch=250, epochs=10,
          validation_data=val_dataset_generator, validation_steps=5)






