from dl.models.unet import mouse_lung_seg
from dl.losses.jaccard import jaccard_distance, jaccard_distance_loss, soft_dice_loss
from dl.generators import data_prep_train_on_batch
from keras.optimizers import Adam
from random import shuffle, sample
import math
import time
import numpy as np
from dl.utils.filemanip import data_split, del_vols
import tensorflow as tf
from keras import backend as K


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def run_batch_all(model, batch_files, s, batch_size):
    files = batch_files[s*batch_size:(s+1)*batch_size]
    x, y = data_prep_train_on_batch(files)
    hist = model.train_on_batch(x, y)
    
    return hist


def run_batch_val_all(model, batch_files, s, batch_size):
    files = batch_files[s*batch_size:(s+1)*batch_size]
    x, y = data_prep_train_on_batch(files)
    hist = model.test_on_batch(x, y)
    
    return hist


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

                                           
def run_training(train_files, validation_files, fold=0):
    
    n_epochs = 100
    training_bs = 41
    validation_bs = 40
    training_steps = math.ceil(len(train_files)/training_bs)
    validation_steps = math.ceil(len(validation_files)/validation_bs)
#     training_steps = 600
#     validation_steps = 500
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
        model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=[jaccard_distance])

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
            hist = run_batch_all(model, train_files, ts, training_bs)
            training_loss.append(hist[0])
            training_jd.append(hist[1])
            if ts in validation_index:
                hist = run_batch_val_all(model, validation_files, vs, validation_bs)
                validation_loss.append(hist[0])
                validation_jd.append(hist[1])
                vs = vs+1
                
        all_loss_training.append(np.mean(training_loss))
        all_loss_val.append(np.mean(validation_loss))
        print('Training and validation for epoch {} ended!\n'.format(str(e+1)))
        print('Training loss: {0}. Jaccard distance: {1}'.format(np.mean(training_loss), np.mean(training_jd)))
        print('Validation loss: {0}. Jaccard distance: {1}'.format(np.mean(validation_loss), np.mean(validation_jd)))
        
        if e == 0:
            weight_name = 'double_feat_per_layer_cross_ent_fold_{0}.h5'.format(fold)
            print('Saving network weights...')
            model.save_weights(weight_name)
        elif (e > 0 and (all_loss_val[e] < np.min(all_loss_val[:-1]))
                and (all_loss_training[e] < np.min(all_loss_training[:-1]))):
            patience = 0
            weight_name = 'double_feat_per_layer_cross_ent_fold_{0}.h5'.format(fold)
            print('Saving network weights...')
            model.save_weights(weight_name)
        elif (e >= 0 and ((all_loss_val[e] >= np.min(all_loss_val[:-1]))
                or (all_loss_training[e] >= np.min(all_loss_training[:-1]))) and patience < 10):
            print('No validation loss improvement with respect to the previous epochs. Weights will not be saved.')
        elif (e >= 0 and ((all_loss_val[e] >= np.min(all_loss_val[:-1]))
                or (all_loss_training[e] >= np.min(all_loss_training[:-1]))) and patience >= 10):
            print('No validation loss improvement with respect to the previous epochs in the last 10 iterations.\n'
                  'Training will be stopped.')
            break
        patience = patience+1
        K.clear_session()
    
    np.savetxt('Training_loss_fold_{}.txt'.format(fold), np.asarray(all_loss_training))
    np.savetxt('Validation_loss_fold_{}.txt'.format(fold), np.asarray(all_loss_val))

start = time.perf_counter()

data_dir = '/mnt/sdb/mouse_data_prep_new/'
save_test_set = True
split = True
len_train = 1000  #102668

for i in range(5):
    print('\nStarting cross-validation: fold {}\n'.format(i+1))
    train_files, validation_files, len_train = data_split(
        data_dir, save_test_set=save_test_set, split=split, train_len=len_train)
    if i == 0:
        print('Number of training files: {}'.format(len(train_files)))
        print('Number of validation files: {}'.format(len(validation_files)))
    run_training(train_files, validation_files, fold=i+1)
    split = False
    save_test_set = False

del_vols(data_dir)

stop = time.perf_counter()
print('\nAll done!'),
print('Elapsed time: {} seconds'.format(int(stop-start)))
