from dl.models.unet import mouse_lung_seg
from dl.losses.jaccard import jaccard_distance_loss
from dl.generators import data_prep_train_on_batch
from keras.optimizers import Adam
import glob
from random import shuffle, sample
import math
import time
import numpy as np
from dl.generators import image_generator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


def dice_loss(smooth):
  def dice(y_true, y_pred):
    return jaccard_distance_loss(y_true, y_pred, smooth)
  return dice


start = time.perf_counter()

data_dir = '/home/fsforazz/Desktop/mouse_nifti'
train_files = (sorted(glob.glob(data_dir+'/training_nifti_2/Mouse*.nii.gz')))#[:102000])
validation_files = (sorted(glob.glob(data_dir+'/validation_nifti_2/Mouse*.nii.gz')))#[:25700])

n_epochs = 15
training_bs = 51
validation_bs = 50
training_steps = math.ceil(len(train_files)/training_bs)
validation_steps = math.ceil(len(validation_files)/validation_bs)
lr = 2e-4
pretrained_weights = None

train_generator = image_generator(train_files, batch_size=50)
validation_generator = image_generator(validation_files, batch_size=50)

weights_name = 'lung_bs=60_spe=172_e=7_loss=bin_crossEntropy_metrics=jacc_dist_whole_dataset.h5'
lr = 2e-4

model_dice = dice_loss(smooth=100)

model = mouse_lung_seg(pretrained_weights=pretrained_weights)
model.compile(optimizer=Adam(lr), loss=jaccard_distance_loss, metrics=['accuracy'])
weight_saver = ModelCheckpoint(weights_name, monitor='val_jaccard_distance_loss', 
                               save_best_only=True, save_weights_only=True)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
hist = model.fit_generator(train_generator,
                           steps_per_epoch=500,
                           validation_data=validation_generator,
                           validation_steps=300,
                           epochs=n_epochs, verbose=2,
                           callbacks = [weight_saver, annealer])

results = model.predict(validation_generator)
