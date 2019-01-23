from models.unet import mouse_lung_seg, mouse_lung_seg_less_feat
from keras.callbacks import LearningRateScheduler
from losses.jaccard import jaccard_distance
from utils.filemanip import data_prep_train_on_batch
from keras.optimizers import Adam
import glob
from random import shuffle


SEED=42

data_dir = '/home/fsforazz/Desktop/mouse_nifti'

train_files = (sorted(glob.glob(data_dir+'/training_nifti_2/Mouse*.nii.gz'))[:102000])
validation_files = (sorted(glob.glob(data_dir+'/validation_nifti_2/Mouse*.nii.gz'))[:25700])


n_epochs = 15
weights_name = 'lung_bs=60_spe=172_e=30_loss=bin_crossEntropy_metrics=jacc_dist_whole_dataset_new.h5'
lr = 2e-4

model = mouse_lung_seg()
model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=[jaccard_distance])

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)

for e in range(n_epochs):

    print('Epoch {}'.format(str(e+1)))
    shuffle(train_files)
    shuffle(validation_files)
    for ts in range(2000):
        files = train_files[ts*51:(ts+1)*51]
        x, y = data_prep_train_on_batch(files)
        training_loss = model.train_on_batch(x, y)
    print('Training loss: {0}. Jaccard distance: {1}'.format(training_loss[0], training_loss[1]))
    for vs in range(514):
        files = validation_files[vs*50:(vs+1)*50]
        x, y = data_prep_train_on_batch(files)
        valdation_loss = model.test_on_batch(x, y)
    print('Validation loss: {0}. Jaccard distance: {1}'.format(valdation_loss[0], valdation_loss[1]))

    model.save_weights('double_feat_per_layer_epoch_{}.h5'.format(str(e+1)))
