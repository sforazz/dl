from models.unet import mouse_lung_seg
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from losses.jaccard import jaccard_distance
from utils.filemanip import data_generator
from utils.mouse_segmentation import data_preparation
from keras.optimizers import Adam
import glob
import random


SEED=42

data_dir = '/home/fsforazz/Desktop/mouse_nifti'

images = sorted(glob.glob(data_dir+'/Mouse*.nii*'))
labels = sorted(glob.glob(data_dir+'/Mask*.nii*'))

train_indexs = random.sample(range(len(images)), 1500)
image_indexs = range(len(images))
test_indexs = [x for x in image_indexs if x not in train_indexs]
with open(data_dir+'/images_for_test.txt', 'w') as f:
    for ind in test_indexs:
        f.write(images[ind]+'\n')

pretrained_weights = None
weights_name = 'lung_bs=60_spe=172_e=10_loss=bin_crossEntropy_metrics=jacc_dist.h5'
for i in range(10):
    print('Starting training with batch {}'.format(i+1))
    if i != 0:
        pretrained_weights = weights_name
    images_batch = [images[x] for x in train_indexs[i*150:(i+1)*150]]
    labels_batch = [labels[x] for x in train_indexs[i*150:(i+1)*150]]
    train_X,valid_X,train_ground,valid_ground = data_preparation(images_batch, labels=labels_batch)

    model = mouse_lung_seg(pretrained_weights=pretrained_weights)
    model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[jaccard_distance])
    weight_saver = ModelCheckpoint(weights_name, monitor='val_jaccard_distance', 
                                   save_best_only=True, save_weights_only=True)
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
    hist = model.fit_generator(data_generator(train_X, train_ground, 60, seed=SEED),
                               steps_per_epoch = 172,
                               validation_data = (valid_X, valid_ground),
                               epochs=10, verbose=2,
                               callbacks = [weight_saver, annealer])

results = model.predict(valid_X)
# saveResult("/home/fsforazz/Desktop/mouse_nifti",results)
