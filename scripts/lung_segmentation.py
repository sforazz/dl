from dl.models.unet import mouse_lung_seg
from dl.losses.jaccard import jaccard_distance, jaccard_distance_loss
from dl.generators import data_prep_train_on_batch
from keras.optimizers import Adam
import glob
from random import shuffle, sample
import math
import time
import numpy as np


def run_batch(batch_files, s, batch_size):

    indexes = sample(range(len(batch_files)), batch_size)
    files = [batch_files[x] for x in indexes]
    x, y = data_prep_train_on_batch(files)
    hist = model.train_on_batch(x, y)
    
    return hist


def run_batch_val(batch_files, s, batch_size):

    indexes = sample(range(len(batch_files)), batch_size)
    files = [batch_files[x] for x in indexes]
    x, y = data_prep_train_on_batch(files)
    hist = model.test_on_batch(x, y)
    
    return hist

start = time.perf_counter()

data_dir = '/home/fsforazz/Desktop/mouse_nifti'
train_files = (sorted(glob.glob(data_dir+'/training_nifti_2/Mouse*.nii.gz')))#[:102000])
validation_files = (sorted(glob.glob(data_dir+'/validation_nifti_2/Mouse*.nii.gz')))#[:25700])

n_epochs = 100
training_bs = 51
validation_bs = 50
# training_steps = math.ceil(len(train_files)/training_bs)
# validation_steps = math.ceil(len(validation_files)/validation_bs)
training_steps = 200
validation_steps = 150
lr = 2e-4

model = mouse_lung_seg()
model.compile(optimizer=Adam(lr), loss=jaccard_distance_loss, metrics=['accuracy'])


for e in range(n_epochs):

    print('\nEpoch {}'.format(str(e+1)))
    shuffle(train_files)
    shuffle(validation_files)

    training_loss = []
    training_jd = []
    validation_loss = []
    validation_jd = []

    validation_index = sample(range(10, training_steps), validation_steps)
    vs = 0

    print('\nTraining and validation started...\n')
    for ts in range(training_steps):
        print('Batch {0}/{1}'.format(ts+1, training_steps), end="\r")
        hist = run_batch(train_files, ts, training_bs)
        training_loss.append(hist[0])
        training_jd.append(hist[1])
        if ts in validation_index:
            hist = run_batch_val(validation_files, vs, validation_bs)
            validation_loss.append(hist[0])
            validation_jd.append(hist[1])
            vs = vs+1
            

    print('Training and validation for epoch {} ended!\n'.format(str(e+1)))
    print('Training loss: {0}. Jaccard distance: {1}'.format(np.mean(training_loss), np.mean(training_jd)))
    print('Validation loss: {0}. Jaccard distance: {1}'.format(np.mean(validation_loss), np.mean(validation_jd)))

    print('Saving network weights...')
    model.save_weights('double_feat_per_layer_epoch_{}_new.h5'.format(str(e+1)))

stop = time.perf_counter()
print('\nAll done!'),
print('Elapsed time: {} seconds'.format(int(stop-start)))
