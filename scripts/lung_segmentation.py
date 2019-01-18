from models.unet import mouse_lung_seg, UNet
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from losses.jaccard import jaccard_distance
from utils.filemanip import data_generator, image_generator
from utils.mouse_segmentation import data_preparation
from keras.optimizers import Adam
import glob
import random


SEED=42

data_dir = '/home/fsforazz/Desktop/mouse_nifti'

images = sorted(glob.glob(data_dir+'/Mouse*.nii*'))
labels = sorted(glob.glob(data_dir+'/Mask*.nii*'))

train_files = sorted(glob.glob(data_dir+'/training/*.png'))[:103000]
validation_files = sorted(glob.glob(data_dir+'/validation/*.png'))[:25700]

# train_indexs = random.sample(range(len(images)), 1500)
# image_indexs = range(len(images))
# test_indexs = [x for x in image_indexs if x not in train_indexs]
# with open(data_dir+'/images_for_test.txt', 'w') as f:
#     for ind in test_indexs:
#         f.write(images[ind]+'\n')
n_epochs = 7
pretrained_weights = None
weights_name = 'lung_bs=60_spe=172_e=7_loss=bin_crossEntropy_metrics=jacc_dist_whole_dataset.h5'
lr = 2e-4

train_generator = image_generator(train_files, batch_size=103)
validation_generator = image_generator(validation_files, batch_size=50)

model = mouse_lung_seg(pretrained_weights=pretrained_weights)
model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=[jaccard_distance])
weight_saver = ModelCheckpoint(weights_name, monitor='val_jaccard_distance', 
                               save_best_only=True, save_weights_only=True)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
hist = model.fit_generator(train_generator,
                           steps_per_epoch=1000,
                           validation_data=validation_generator,
                           validation_steps=514,
                           epochs=n_epochs, verbose=2,
                           callbacks = [weight_saver, annealer])
# for i in range(10):
#     print('Starting training with batch {}'.format(i+1))
#     if i != 0:
#         pretrained_weights = weights_name
#         n_epochs = 2
#         lr = lr/2
#     images_batch = [images[x] for x in train_indexs[i*150:(i+1)*150]]
#     labels_batch = [labels[x] for x in train_indexs[i*150:(i+1)*150]]
#     train_X,valid_X,train_ground,valid_ground = data_preparation(images_batch, labels=labels_batch)
# 
#     model = mouse_lung_seg(pretrained_weights=pretrained_weights)
# #     if i != 0:
# #         
#     model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=[jaccard_distance])
#     weight_saver = ModelCheckpoint(weights_name, monitor='val_jaccard_distance', 
#                                    save_best_only=True, save_weights_only=True)
#     annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
#     hist = model.fit_generator(data_generator(train_X, train_ground, 60, seed=SEED),
#                                steps_per_epoch = 172,
#                                validation_data = (valid_X, valid_ground),
#                                epochs=n_epochs, verbose=2,
#                                callbacks = [weight_saver, annealer])

results = model.predict(validation_generator)
# saveResult("/home/fsforazz/Desktop/mouse_nifti",results)