from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from .utils.filemanip import get_nifti
from .utils.mouse_segmentation import preprocessing


def data_generator(x_train, y_train, batch_size, seed=42):

    data_generator = ImageDataGenerator().flow(x_train, x_train, batch_size, seed=seed)
    mask_generator = ImageDataGenerator().flow(y_train, y_train, batch_size, seed=seed)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()

        yield x_batch, y_batch


def image_generator(files, batch_size = 64):
    
    while True:
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a = files, size = batch_size)
        batch_input = []
        batch_output = [] 
        
        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
#             with open('/home/fsforazz/Desktop/used_images.txt', 'a') as f:
#                 f.write(input_path+'\n')
            image = get_nifti(input_path)
            mask = get_nifti(input_path, labels=True)
          
#             image = preprocess_input(image.astype('float64'), mode='tf')
            image = preprocessing(image)
            mask = preprocessing(mask, label=True)
            batch_input += [image]
            batch_output += [mask]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        
        yield(batch_x, batch_y)


def data_prep_train_on_batch(files):
    
    images = []
    labels = []

    for file in files:

        image = get_nifti(file)
        mask = get_nifti(file, labels=True)

        image = preprocessing(image)
        mask = preprocessing(mask, label=True)
        images += [image]
        labels += [mask]
    
    return np.array(images), np.array(labels)
