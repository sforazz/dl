"""
Train a UNET model to predict a continuous 3D image from a given
3D continuous brain image.

The example here uses the input image as a target image (aka an 'Autoencoder') but the
target image can be any other brain image.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

from keras import callbacks as cbks

# base_dir = '/mnt/sdb/BRATS2015_Training/HGG/single_patch_data/'
base_dir = '/data/data_segmentation/'
#os.chdir('/home/fsforazz/git/Unet-ants/code')

# local imports
# from dl.sampling import DataLoader, CSVDataset
from dl.sampling.dataloader import DataLoader
from dl.sampling.datasets import CSVDataset
from dl.sampling import transforms as tx
from dl.models.unet import create_unet_model3D, unet_model_3d


data_dir = base_dir
results_dir = base_dir+'results_3D/'
try:
    os.mkdir(results_dir)
except:
    pass

fn = lambda im: im.swapaxes(0,-1)
# tx.Compose lets you string together multiple transforms
co_tx = tx.Compose([tx.TypeCast('float32')])

input_tx = tx.LambdaTransform(fn) # scale between -1 and 1

target_tx = tx.Compose([tx.LambdaTransform(fn),
                        tx.OneHot()]) # convert segmentation image to One-Hot representation for cross-entropy loss
# target_tx = tx.LambdaTransform(fn)
# use a co-transform, meaning the same transform will be applied to input+target images at the same time 
# this is necessary since Affine transforms have random parameter draws which need to be shared
dataset = CSVDataset(filepath=data_dir+'image_filemap.csv',
                     base_path='/data/data_segmentation/', # this path will be appended to all of the filenames in the csv file
                     input_cols=['Images'], # column in dataframe corresponding to inputs (can be an integer also)
                     target_cols=['Segmentations'],
                     target_transform=target_tx,
                     input_transform=input_tx) # run co transforms before input/target transforms


# split into train and test set based on the `train-test` column in the csv file
# this splits alphabetically by values, and since 'test' comes before 'train' thus val_data is returned before train_data
val_data, train_data = dataset.split_by_column('TrainTest')

# create a dataloader .. this is basically a keras DataGenerator -> can be fed to `fit_generator`
batch_size = 10
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# write an example batch to a folder as JPEG
#train_loader.write_a_batch(data_dir+'example_batch/')

n_labels = train_data[0][1].shape[0]
# create model
# model = create_unet_model3D(input_image_size=train_data[0][0].shape, n_labels=n_labels, layers=4,
#                             mode='classification')
model = unet_model_3d(train_data[0][0].shape)

callbacks = [cbks.ModelCheckpoint(results_dir+'segmentation-weights.h5', monitor='val_loss', save_best_only=True),
            cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1)]

model.fit_generator(generator=iter(train_loader), steps_per_epoch=np.ceil(len(train_data)/batch_size), 
                    epochs=100, verbose=1, callbacks=callbacks, 
                    shuffle=True, 
                    validation_data=iter(val_loader), validation_steps=np.ceil(len(val_data)/batch_size), 
                    class_weight=None, max_queue_size=10, 
                    workers=1, use_multiprocessing=False,  initial_epoch=0)


### RUNNING INFERENCE ON THE NON-AUGMENTED DATA

# load all the validation data into memory.. not at all necessary but easier for this example
#val_x, val_y = val_data.load()
#real_val_x, real_val_y = val_data.load()
#real_val_y_pred = model.predict(real_val_x)



