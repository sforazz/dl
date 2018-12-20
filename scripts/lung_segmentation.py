from models.unet import *
import nibabel as nib
from sklearn.model_selection import train_test_split
import glob
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
SEED=42


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def data_preparation(data_dir):
    
    images = sorted(glob.glob(data_dir+'/Mouse*.nii*'))
    labels = sorted(glob.glob(data_dir+'/Mask*.nii*'))
    if len(images) != len(labels):
        raise Exception('Different number of images and labels.')
    
    images = [nib.load(x).get_data()[:, :, i] for x in images[:100]
              for i in range(nib.load(x).get_data().shape[2])]
    labels = [nib.load(x).get_data()[:, :, i] for x in labels[:100]
              for i in range(nib.load(x).get_data().shape[2])]
    images = np.asarray(images)
    labels = np.asarray(labels)
    
    images = images.reshape(-1, 86, 86, 1)
    labels = labels.reshape(-1, 86, 86, 1)
    
    images = (images-np.mean(images))/(np.std(images))
    
    temp = np.zeros([images.shape[0], 96, 96, 1])
    temp[:,10:,10:,:] = images
    images = temp
    temp1 = np.zeros([images.shape[0], 96, 96, 1])
    temp1[:,10:,10:,:] = labels
    labels = temp1
    train_X,valid_X,train_ground,valid_ground = train_test_split(images,
                                                                 labels,
                                                                 test_size=0.2,
                                                                 random_state=13)
    
    return train_X,valid_X,train_ground,valid_ground
    
def my_generator(x_train, y_train, batch_size):

    data_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


data_dir = '/home/fsforazz/Desktop/mouse_nifti' 
train_X,valid_X,train_ground,valid_ground = data_preparation(data_dir)
image_batch, mask_batch = next(my_generator(train_X, train_ground, 8))

# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')
# mouse_data = trainGenerator(6,'/home/fsforazz/Desktop/mouse_nifti','Mouse','Mask',{},save_to_dir = None,
#                             target_size=(86, 86, 86))

model = mouse_lung_seg()
model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
weight_saver = ModelCheckpoint('lung.h5', monitor='val_dice_coef', 
                                              save_best_only=True, save_weights_only=True)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
hist = model.fit_generator(my_generator(train_X, train_ground, 30),
                           steps_per_epoch = 200,
                           validation_data = (valid_X, valid_ground),
                           epochs=10, verbose=2,
                           callbacks = [weight_saver])
# model_checkpoint = ModelCheckpoint('unet_fibrosis.hdf5', monitor='loss',verbose=1, save_best_only=True)
# model.fit(train_X, train_ground, batch_size=None, steps_per_epoch=1,epochs=150,callbacks=[model_checkpoint],validation_data=(valid_X, valid_ground), validation_steps=50)
# 
# results = model.predict(valid_X)
# saveResult("/home/fsforazz/Desktop/mouse_nifti",results)
