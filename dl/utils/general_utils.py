import os
import tensorflow as tf
from scipy import ndimage
import numpy as np


def remove_files(files):
    """
    Remove files from disk
    args: files (str or list) remove all files in 'files'
    """

    if isinstance(files, (list, tuple)):
        for f in files:
            if os.path.isfile(os.path.expanduser(f)):
                os.remove(f)
    elif isinstance(files, str):
        if os.path.isfile(os.path.expanduser(files)):
            os.remove(files)


def create_dir(dirs):
    """
    Create directory
    args: dirs (str or list) create all dirs in 'dirs'
    """

    if isinstance(dirs, (list, tuple)):
        for d in dirs:
            if not os.path.exists(os.path.expanduser(d)):
                os.makedirs(d)
    elif isinstance(dirs, str):
        if not os.path.exists(os.path.expanduser(dirs)):
            os.makedirs(dirs)


def setup_logging(model_name, logging_dir='../../'):
    
    # Output path where we store experiment log and weights
    model_dir = os.path.join(logging_dir, 'models', model_name)

    fig_dir = os.path.join(logging_dir, 'figures')
    
    # Create if it does not exist
    create_dir([model_dir, fig_dir])
    

def write_log(callback, name, loss, batch_no):
    """
    Write training summary to TensorBoard
    """
    # for name, value in zip(names, logs):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = loss
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


def normalize_array_max(array):
    max_value = max(array.flatten())
    if max_value > 0:
        array = array / max_value
        array = (array - 0.5)*2
    return array, max_value


def sobel_3D(image):
    
    if len(image.shape) > 3:
        sob = np.zeros((list(image.shape[:-1])+[3]), dtype=np.float16)
        for i in range(image.shape[0]):
            sx = ndimage.sobel(image[i, :, :, :, 0], axis=0, mode='constant')
            sy = ndimage.sobel(image[i, :, :, :, 0], axis=1, mode='constant')
            sz = ndimage.sobel(image[i, :, :, :, 0], axis=2, mode='constant')
            sx = sx.reshape((image.shape[1], image.shape[2], image.shape[3], 1))
            sy = sy.reshape((image.shape[1], image.shape[2], image.shape[3], 1))
            sz = sz.reshape((image.shape[1], image.shape[2], image.shape[3], 1))
            concat = np.concatenate([sx, sy, sz], axis=-1)
            sob[i, :, :, :, :] = concat
    else:         
        sx = ndimage.sobel(image, axis=0, mode='constant')
        sy = ndimage.sobel(image, axis=1, mode='constant')
        sz = ndimage.sobel(image, axis=2, mode='constant')
        sx = sx.reshape((image.shape[0], image.shape[1], image.shape[2], 1))
        sy = sy.reshape((image.shape[0], image.shape[1], image.shape[2], 1))
        sz = sz.reshape((image.shape[0], image.shape[1], image.shape[2], 1))
        sob = np.concatenate([sx, sy, sz], axis=-1)
    return sob
