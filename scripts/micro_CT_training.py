from dl.models.unet import mouse_lung_seg
from dl.losses.jaccard import jaccard_distance, jaccard_distance_loss, soft_dice_loss
from dl.generators import data_prep_train_on_batch, load_data_2D
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


def run_batch_all(model, batch_files, train_masks, s, batch_size):
#     files = batch_files[s*batch_size:(s+1)*batch_size]
    x = batch_files[s*batch_size:(s+1)*batch_size]
    y = train_masks[s*batch_size:(s+1)*batch_size]
#     x = load_data_2D('', '', files, extract_edges=False)
#     y = load_data_2D('', '', masks, extract_edges=False, normalize=False)
#     x, y = data_prep_train_on_batch(files, method='human')
    hist = model.train_on_batch(x, y)
    
    return hist


def run_batch_val_all(model, batch_files, train_masks, s, batch_size):
#     files = batch_files[s*batch_size:(s+1)*batch_size]
    x = batch_files[s*batch_size:(s+1)*batch_size]
    y = train_masks[s*batch_size:(s+1)*batch_size]
#     x = load_data_2D('', '', files, extract_edges=False)
#     y = load_data_2D('', '', masks, extract_edges=False, normalize=False)
#     x, y = data_prep_train_on_batch(files, method='human')
    hist = model.test_on_batch(x, y)
    
    return hist


def run_batch(model, batch_files, train_masks, s, batch_size):

    indexes = sample(range(len(batch_files)), batch_size)
    x = np.asarray([batch_files[x, :, :, :] for x in indexes])
    y = np.asarray([train_masks[x, :, :, :] for x in indexes])
#     x, y = data_prep_train_on_batch(files)
    hist = model.train_on_batch(x, y)
    
    return hist


def run_batch_val(model, batch_files, train_masks, s, batch_size):

    indexes = sample(range(len(batch_files)), batch_size)
    x = np.asarray([batch_files[x, :, :, :] for x in indexes])
    y = np.asarray([train_masks[x, :, :, :] for x in indexes])
#     x, y = data_prep_train_on_batch(files)
    hist = model.test_on_batch(x, y)
    
    return hist


def data_split(data_dir, train_len=0):
    
    data = []
    masks = []

    for root, _, files in os.walk(data_dir): 
        for name in files: 
            if name.endswith('.nii.gz') and 'mask' not in name and 'vol' in name: 
                data.append(os.path.join(root, name))
            elif name.endswith('.nii.gz') and 'mask' in name and 'vol' in name: 
                masks.append(os.path.join(root, name))

    data = sorted(data)
    masks = sorted(masks)

    training_indxs = sample(range(len(data)), train_len)
    validation_indxs = [x for x in range(len(data)) if x not in training_indxs]

    train_files = [data[x] for x in training_indxs]
    validation_files = [data[x] for x in validation_indxs]
    train_masks = [masks[x] for x in training_indxs]
    validation_masks = [masks[x] for x in validation_indxs]
    train_files = load_data_2D('', '', train_files, extract_edges=False)
    validation_files = load_data_2D('', '', validation_files, extract_edges=False)
    train_masks = load_data_2D('', '', train_masks, extract_edges=False, normalize=False)
    validation_masks = load_data_2D('', '', validation_masks, extract_edges=False, normalize=False)
    
    return train_files, validation_files, len(train_files), train_masks, validation_masks


def run_training(train_files, train_masks, validation_files, validation_masks, fold=0, w=None):
    
#     data_dir = '/home/fsforazz/Desktop/mouse_nifti'
#     train_files = (sorted(glob.glob(data_dir+'/training_nifti_2/Mouse*.nii.gz')))#[:102000])
#     validation_files = (sorted(glob.glob(data_dir+'/validation_nifti_2/Mouse*.nii.gz')))#[:25700])
    
    n_epochs = 30
    training_bs = 41
    validation_bs = 40
    training_steps = math.ceil(len(train_files)/training_bs)
    validation_steps = math.ceil(len(validation_files)/validation_bs)
#     training_steps = 500
#     validation_steps = 400
    lr_0 = 2e-4
    
    model = mouse_lung_seg()
    for layer in model.layers[:26]:
        layer.trainable=False
    
    all_loss_training = []
    all_loss_val = []
    patience = 0
    if w is None:
        weight_name = 'double_feat_per_layer_cross_ent_fold_{0}_human_CT.h5'.format(fold)
    else:
        weight_name = w
    for e in range(n_epochs):
    
        print('\nEpoch {}'.format(str(e+1)))

        if e >= 0:
            model = mouse_lung_seg(pretrained_weights=weight_name)
            for layer in model.layers[:26]:
                layer.trainable=False
        lr = lr_0 * 0.99**e
        model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=[jaccard_distance])

#         shuffle(train_files)
#         shuffle(validation_files)
    
        training_loss = []
        training_jd = []
        validation_loss = []
        validation_jd = []
    
        validation_index = sample(range(10, training_steps), validation_steps)
        vs = 0
    
        print('\nTraining and validation started...\n')
        for ts in range(training_steps):
            print('Batch {0}/{1}'.format(ts+1, training_steps), end="\r")
            hist = run_batch_all(model, train_files, train_masks, ts, training_bs)
            training_loss.append(hist[0])
            training_jd.append(hist[1])
            if ts in validation_index:
                hist = run_batch_val_all(model, validation_files, validation_masks, vs, validation_bs)
                validation_loss.append(hist[0])
                validation_jd.append(hist[1])
                vs = vs+1
                
        all_loss_training.append(np.mean(training_loss))
        all_loss_val.append(np.mean(validation_loss))
        print('Training and validation for epoch {} ended!\n'.format(str(e+1)))
        print('Training loss: {0}. Jaccard distance: {1}'.format(np.mean(training_loss), np.mean(training_jd)))
        print('Validation loss: {0}. Jaccard distance: {1}'.format(np.mean(validation_loss), np.mean(validation_jd)))
        
        if e == 0:
#             weight_name = 'double_feat_per_layer_cross_ent_fold_{0}.h5'.format(fold)
            print('Saving network weights...')
            model.save_weights(weight_name)
        elif (e > 0 and all_loss_training[e] < np.min(all_loss_training[:-1])):
            patience = 0
#             weight_name = 'double_feat_per_layer_cross_ent_fold_{0}.h5'.format(fold)
            print('Saving network weights...')
            model.save_weights(weight_name)
        elif (e >= 0 and all_loss_training[e] >= np.min(all_loss_training[:-1]) and patience < 10):
            print('No validation loss improvement with respect to the previous epochs. Weights will not be saved.')
            print('Patience: {}'.format(patience))
        elif  (e >= 0 and all_loss_training[e] >= np.min(all_loss_training[:-1]) and patience >= 10):
            print('No validation loss improvement with respect to the previous epochs in the last 10 iterations.\n'
                  'Training will be stopped.')
            break
        patience = patience+1
        K.clear_session()
    
    np.savetxt('Training_loss_fold_{}_micro_CT.txt'.format(fold), np.asarray(all_loss_training))
    np.savetxt('Validation_loss_fold_{}_micro_CT.txt'.format(fold), np.asarray(all_loss_val))

start = time.perf_counter()

data_dir = '/mnt/sdb/results_micro_CT/training/'
len_train = 700  #102668
weights = ['/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/double_feat_per_layer_cross_ent_fold_1.h5',
           '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/double_feat_per_layer_cross_ent_fold_2.h5',
           '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/double_feat_per_layer_cross_ent_fold_3.h5',
           '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/double_feat_per_layer_cross_ent_fold_4.h5',
           '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/double_feat_per_layer_cross_ent_fold_5.h5']

for i in range(5):
    print('\nStarting cross-validation: fold {}\n'.format(i+1))
    try:
        train_files = np.load('/mnt/sdb/results_micro_CT/training/train_files.npy')
        validation_files = np.load('/mnt/sdb/results_micro_CT/training/validation_files.npy')
        train_masks = np.load('/mnt/sdb/results_micro_CT/training/train_masks.npy')
        validation_masks = np.load('/mnt/sdb/results_micro_CT/training/validation_masks.npy')
        len_train = train_files.shape[0]
        if i > 0:
            n_swap = sample(range(0, validation_files.shape[0]), 1)[0]
            val_ind = sample(range(0, validation_files.shape[0]), n_swap)
            train_ind = sample(range(0, train_files.shape[0]), n_swap)
            train_files_new = np.copy(train_files)
            validation_files_new = np.copy(validation_files)
            train_masks_new = np.copy(train_masks)
            validation_masks_new = np.copy(validation_masks)
            for ind in range(len(train_ind)):
                train_files_new[train_ind[ind], :, :, 0] = validation_files[val_ind[ind], :, :, 0]
                validation_files_new[val_ind[ind], :, :, 0] = train_files[train_ind[ind], :, :, 0]
                train_masks_new[train_ind[ind], :, :, 0] = validation_masks[val_ind[ind], :, :, 0]
                validation_masks_new[val_ind[ind], :, :, 0] = train_masks[train_ind[ind], :, :, 0]
            train_files = train_files_new
            validation_files = validation_files_new
            train_masks = train_masks_new
            validation_masks = validation_masks_new
    except:
        train_files, validation_files, len_train, train_masks, validation_masks = data_split(
            data_dir, train_len=len_train)
        np.save('/mnt/sdb/results_micro_CT/training/train_files.npy', train_files)
        np.save('/mnt/sdb/results_micro_CT/training/validation_files.npy', validation_files)
        np.save('/mnt/sdb/results_micro_CT/training/train_masks.npy', train_masks)
        np.save('/mnt/sdb/results_micro_CT/training/validation_masks.npy', validation_masks)
    if i == 0:
        print('Number of training files: {}'.format(len(train_files)))
        print('Number of validation files: {}'.format(len(validation_files)))
    run_training(train_files, train_masks, validation_files, validation_masks, fold=i+1, w=weights[i])

stop = time.perf_counter()
print('\nAll done!'),
print('Elapsed time: {} seconds'.format(int(stop-start)))
