from dl.models.unet import mouse_lung_seg
from dl.losses.jaccard import jaccard_distance, jaccard_distance_loss, soft_dice_loss
from dl.generators import data_prep_train_on_batch
from keras.optimizers import Adam
import glob
from random import shuffle, sample
import math
import time
import numpy as np
import os
from dl.utils.filemanip import split_nifti
import tensorflow as tf
from keras import backend as K


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def run_batch(model, batch_files, s, batch_size):

    indexes = sample(range(len(batch_files)), batch_size)
    files = [batch_files[x] for x in indexes]
    x, y = data_prep_train_on_batch(files)
    hist = model.train_on_batch(x, y)
    
    return hist


def run_batch_val(model, batch_files, s, batch_size):

    indexes = sample(range(len(batch_files)), batch_size)
    files = [batch_files[x] for x in indexes]
    x, y = data_prep_train_on_batch(files)
    hist = model.test_on_batch(x, y)
    
    return hist

def data_split(data_dir, save_test_set=True, split=True, train_len=0):
    
    data = []
    masks = []
    train_files = []
    validation_files = []

    if split:
        for root, _, files in os.walk(data_dir): 
            for name in files: 
                if name.endswith('.nii.gz') and 'Raw_data' in name: 
                    data.append(os.path.join(root, name))
                elif name.endswith('.nii.gz') and 'Raw_data' not in name: 
                    masks.append(os.path.join(root, name))
    else:
        for root, _, files in os.walk(data_dir): 
            for name in files: 
                if name.endswith('.nii.gz') and 'Raw_data' in name and 'vol_' in name: 
                    data.append(os.path.join(root, name))
                elif name.endswith('.nii.gz') and 'Raw_data' not in name and 'vol_' in name: 
                    masks.append(os.path.join(root, name))

    data = sorted(data)
    masks = sorted(masks)

    if save_test_set:
        test_set_indxs = sample(range(len(data)), 169)
        test_set = [data[x] for x in test_set_indxs]
        np.savetxt(data_dir+'/Test_files.txt', np.asarray(test_set), fmt='%s')
        training_indxs = [x for x in range(len(data)) if x not in test_set_indxs]
        training_indxs = sample(training_indxs, train_len)
        validation_indxs = [x for x in range(len(data)) if x not in training_indxs+test_set_indxs]
    else:
        training_indxs = sample(range(len(data)), train_len)
        validation_indxs = [x for x in range(len(data)) if x not in training_indxs]

    if split:
        print('\nSplitting the NIFTI files along the axial direction...\n')
        for index in training_indxs:
            files = split_nifti(data[index])
            _ = split_nifti(masks[index])
            train_files = train_files+files
    
        for index in validation_indxs:
            files = split_nifti(data[index])
            _ = split_nifti(masks[index])
            validation_files = validation_files+files
        print('Splitting done!\n')
    else:
        train_files = [data[x] for x in training_indxs]
        validation_files = [data[x] for x in validation_indxs]
    
    return train_files, validation_files, len(train_files)


def del_vols(data_dir):
    
    data = []
    for root, _, files in os.walk(data_dir): 
        for name in files: 
            if name.endswith('.nii.gz') and 'vol_' in name: 
                data.append(os.path.join(root, name))
    
    for f in data:
        os.remove(f)

                                           
def run_training(train_files, validation_files, fold=0):
    
#     data_dir = '/home/fsforazz/Desktop/mouse_nifti'
#     train_files = (sorted(glob.glob(data_dir+'/training_nifti_2/Mouse*.nii.gz')))#[:102000])
#     validation_files = (sorted(glob.glob(data_dir+'/validation_nifti_2/Mouse*.nii.gz')))#[:25700])
    
    n_epochs = 100
    training_bs = 40
    validation_bs = 40
    # training_steps = math.ceil(len(train_files)/training_bs)
    # validation_steps = math.ceil(len(validation_files)/validation_bs)
    training_steps = 400
    validation_steps = 300
    lr_0 = 2e-4
    
    model = mouse_lung_seg()
    
    all_loss_training = []
    all_loss_val = []
    patience = 0
    weight_name = None
    for e in range(n_epochs):
    
        print('\nEpoch {}'.format(str(e+1)))

        if e > 0:
            model = mouse_lung_seg(pretrained_weights=weight_name)
        lr = lr_0 * 0.99**e
        model.compile(optimizer=Adam(lr), loss=jaccard_distance_loss, metrics=['accuracy'])

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
            hist = run_batch(model, train_files, ts, training_bs)
            training_loss.append(hist[0])
            training_jd.append(hist[1])
            if ts in validation_index:
                hist = run_batch_val(model, validation_files, vs, validation_bs)
                validation_loss.append(hist[0])
                validation_jd.append(hist[1])
                vs = vs+1
                
        all_loss_training.append(np.mean(training_loss))
        all_loss_val.append(np.mean(validation_loss))
        print('Training and validation for epoch {} ended!\n'.format(str(e+1)))
        print('Training loss: {0}. Jaccard distance: {1}'.format(np.mean(training_loss), np.mean(training_jd)))
        print('Validation loss: {0}. Jaccard distance: {1}'.format(np.mean(validation_loss), np.mean(validation_jd)))
        
        if e == 0:
            weight_name = 'double_feat_per_layer_fold_{0}.h5'.format(fold)
            print('Saving network weights...')
            model.save_weights(weight_name)
        elif (e > 0 and (all_loss_val[e] < np.min(all_loss_val[:-1]))
                and (all_loss_training[e] < np.min(all_loss_training[:-1]))):
            patience = 0
            weight_name = 'double_feat_per_layer_fold_{0}.h5'.format(fold)
            print('Saving network weights...')
            model.save_weights(weight_name)
        elif (e >= 0 and (all_loss_val[e] >= np.min(all_loss_val[:-1]))
                and (all_loss_training[e] >= np.min(all_loss_training[:-1])) and patience < 10):
            print('No validation loss improvement with respect to the previous epochs. Weights will not be saved.')
        elif (e >= 0 and (all_loss_val[e] >= np.min(all_loss_val[:-1]))
                and (all_loss_training[e] >= np.min(all_loss_training[:-1])) and patience >= 10):
            print('No validation loss improvement with respect to the previous epochs in the last 10 iterations.\n'
                  'Training will be stopped.')
            break
        patience = patience+1
        K.clear_session()
    
    np.savetxt('Training_loss_fold_{}.txt'.format(fold), np.asarray(all_loss_training))
    np.savetxt('Validation_loss_fold_{}.txt'.format(fold), np.asarray(all_loss_val))

start = time.perf_counter()

data_dir = '/mnt/sdb/mouse_data_prep_new/'
save_test_set = False
split = False
len_train = 102668 #1200  #102668

for i in range(5):
    print('\nStarting cross-validation: fold {}\n'.format(i+1))
    train_files, validation_files, len_train = data_split(
        data_dir, save_test_set=save_test_set, split=split, train_len=len_train)
    run_training(train_files, validation_files, fold=i+1)
    split = False
    save_test_set = False

del_vols(data_dir)

stop = time.perf_counter()
print('\nAll done!'),
print('Elapsed time: {} seconds'.format(int(stop-start)))
