from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from tensorflow.python.keras.applications.resnet50 import preprocess_input
import numpy as np
import os


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


def get_image(image_path, labels=False):
    
    if labels:
        path, filename = os.path.split(image_path)
        img_number = filename.split('.')[0].split('_')[1]
        mask_path = os.path.join(path, 'Mask_{}.png'.format(img_number))
        img = imread(mask_path)
    else:
        img = imread(image_path)
    
    return(img)


def image_generator(files, batch_size = 64):
    
    while True:
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a = files, size = batch_size)
        batch_input = []
        batch_output = [] 
        
        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            image = get_image(input_path)
            mask = get_image(input_path, labels=True)
          
            image = preprocess_input(image=image.astype('float64'), mode='tf')
            batch_input += [image]
            batch_output += [mask]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        
        yield(batch_x, batch_y)