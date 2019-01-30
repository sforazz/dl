from dl.models.unet import mouse_lung_seg
from keras.callbacks import LearningRateScheduler
from dl.losses.jaccard import jaccard_distance
from dl.generators import data_prep_train_on_batch
from keras.optimizers import Adam
import glob
from random import shuffle
import math
import time

start = time.perf_counter()

data_dir = '/media/fsforazz/extra_HD/new_workstation/mouse_nifti'
train_files = (sorted(glob.glob(data_dir+'/training_nifti_2/Mouse*.nii.gz')))#[:102000])
validation_files = (sorted(glob.glob(data_dir+'/validation_nifti_2/Mouse*.nii.gz')))#[:25700])

n_epochs = 15
training_bs = 51
validation_bs = 50
training_steps = math.ceil(len(train_files)/training_bs)
validation_steps = math.ceil(len(validation_files)/validation_bs)

weights_name = 'lung_bs=60_spe=172_e=30_loss=bin_crossEntropy_metrics=jacc_dist_whole_dataset_new.h5'
lr = 2e-4

model = mouse_lung_seg()
model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=[jaccard_distance])

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)

for e in range(n_epochs):

    print('\nEpoch {}'.format(str(e+1)))
    shuffle(train_files)
    shuffle(validation_files)

    print('\nTraining started...\n')
    for ts in range(training_steps):
        print('Batch {0}/{1}'.format(ts+1, training_steps), end="\r")
        files = train_files[ts*training_bs:(ts+1)*training_bs]
        x, y = data_prep_train_on_batch(files)
        training_loss = model.train_on_batch(x, y)
    print('Training ended!\n')
    print('Training loss: {0}. Jaccard distance: {1}'.format(training_loss[0], training_loss[1]))

    print('\nValidation started...\n')
    for vs in range(validation_steps):
        print('Batch {0}/{1}'.format(vs+1, validation_steps), end="\r")
        files = validation_files[vs*validation_bs:(vs+1)*validation_bs]
        x, y = data_prep_train_on_batch(files)
        valdation_loss = model.test_on_batch(x, y)
    print('Validation ended!\n')
    print('Validation loss: {0}. Jaccard distance: {1}'.format(valdation_loss[0], valdation_loss[1]))

    print('Saving network weights...')
    model.save_weights('double_feat_per_layer_epoch_{}.h5'.format(str(e+1)))

stop = time.perf_counter()
print('\nAll done!'),
print('Elapsed time: {} seconds'.format(int(stop-start)))
