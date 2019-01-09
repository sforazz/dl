from keras.preprocessing.image import ImageDataGenerator


def data_generator(x_train, y_train, batch_size, seed=42):

    data_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(x_train, x_train, batch_size, seed=seed)
    mask_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(y_train, y_train, batch_size, seed=seed)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()

        yield x_batch, y_batch