from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
import numpy as np
import os
import nibabel as nib
from utils.mouse_segmentation import preprocessing


def batch(iterable, size):
    # For item i in a range that is a length of l,
    for i in range(0, len(iterable), size):
        # Create an index range for l of n items:
        return iterable[i:i+size]


def data_generator(x_train, y_train, batch_size, seed=42):

    data_generator = ImageDataGenerator().flow(x_train, x_train, batch_size, seed=seed)
    mask_generator = ImageDataGenerator().flow(y_train, y_train, batch_size, seed=seed)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()

        yield x_batch, y_batch


def get_png(image_path, labels=False):
    
    if labels:
        path, filename = os.path.split(image_path)
        img_number = filename.split('.')[0].split('_')[1]
        mask_path = os.path.join(path, 'Mask_{}.png'.format(img_number))
        img = imread(mask_path)
    else:
        img = imread(image_path)
    
    return(img)


def get_nifti(image_path, labels=False):
    
    if labels:
        path, filename = os.path.split(image_path)
        img_number = filename.split('.')[0].split('_')[1]
        slice_number = filename.split('.')[0].split('_')[2]
        mask_path = os.path.join(path, 'Mask_{0}_{1}.nii.gz'.format(img_number, slice_number))
        img = nib.load(mask_path).get_data()
    else:
        img = nib.load(image_path).get_data()
    
    return(img)


def image_generator(files, batch_size = 64):
    
    while True:
        # Select files (paths/indices) for the batch
#         if os.path.isfile('/home/fsforazz/Desktop/used_images.txt'):
#             with open('/home/fsforazz/Desktop/used_images.txt', 'r') as f:
#                 used_files = [x.strip() for x in f]
#             if len(used_files) == 127700:
#                 os.remove('/home/fsforazz/Desktop/used_images.txt')
#             else:
#                 files = [f for f in files if f not in used_files]
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