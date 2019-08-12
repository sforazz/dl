import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2
from basecore.utils.filemanip import split_filename
import nrrd


def data_preparation(images, labels=None, preproc='zscore'):
    
    n_images = len(images)
    images = [nib.load(x).get_data()[:, :, i] for x in images
              for i in range(nib.load(x).get_data().shape[2])]
    
    images = np.asarray(images)
    images = images.reshape(-1, 86, 86, 1)
    if preproc == 'zscore':
        images = (images-np.mean(images))/(np.std(images))
    elif preproc == 'zero-one':
        images = images.astype('float64')
        images -= np.min(images)
        images /= (np.max(images)-np.min(images))
    temp = np.zeros([images.shape[0], 96, 96, 1])
    temp[:,10:,10:,:] = images
    images = temp

    if labels is not None:

        if n_images != len(labels):
            raise Exception('Different number of images and labels.')
        
        labels = [nib.load(x).get_data()[:, :, i] for x in labels
                  for i in range(nib.load(x).get_data().shape[2])]
        
        labels = np.asarray(labels)
        labels = labels.reshape(-1, 86, 86, 1)
        temp1 = np.zeros([images.shape[0], 96, 96, 1])
        temp1[:,10:,10:,:] = labels
        labels = temp1
        train_X,valid_X,train_ground,valid_ground = train_test_split(images,
                                                                     labels,
                                                                     test_size=0.2,
                                                                     random_state=13)
    
        return train_X, valid_X, train_ground, valid_ground
    else:
        return images


def save_results(im2save, ref, save_dir=None):
    
#     basedir, basename = os.path.split(ref)
    basedir, basename, ext = split_filename(ref)
    out_basename = basename.split('.')[0]+'_lung_seg'+ext
    if save_dir is not None:
        outname = os.path.join(save_dir, out_basename)
    else:
        outname = os.path.join(basedir, out_basename)
    
    if ext == '.nii' or ext == '.nii.gz':
        ref = nib.load(ref)
        im2save = nib.Nifti1Image(im2save, ref.affine)
        nib.save(im2save, outname)
    elif ext == '.nrrd':
        _, ref_hd = nrrd.read(ref)
        nrrd.write(outname, im2save, header=ref_hd)
    else:
        raise Exception('Extension "{}" is not recognize!'.format(ext))

    return outname


def postprocessing(im2save, method='mouse_fibrosis'):

    if method == 'mouse_fibrosis':
        image = im2save[:, 10:, 10:]
        image = image.reshape(-1, 86, 86)
    elif method == 'gtv':
        image_old = im2save.reshape(-1, 128, 128)
        image = np.zeros((image_old.shape[0], 512, 512))
        for z in range(image.shape[0]):
            image[z, :, :] = cv2.resize(image_old[z, :, :], (512, 512), interpolation=cv2.INTER_AREA)
    elif method=='human':
        image_old = im2save[:, 10:, 10:]
        image_old = image_old.reshape(-1, 86, 86)
        image_old = image_old.reshape(-1, 86, 86)
        image = np.zeros((image_old.shape[0], 512, 512))
        for z in range(image.shape[0]):
            image[z, :, :] = cv2.resize(image_old[z, :, :], (512, 512), interpolation=cv2.INTER_AREA)
        
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 0, 1)

    return image


def preprocessing(image, label=False, method='mouse_fibrosis'):
    
    image = image.astype('float64')
    if method == 'mouse_fibrosis' or method == 'micro_ct':
        image = image.reshape(86, 86, 1)
        if not label:
            image -= np.min(image)
            image /= (np.max(image)-np.min(image))
        temp = np.zeros([96, 96, 1])
        temp[10:,10:,:] = image
        image = temp
    elif method == 'human':
#         image = image[300:386, 300:386]
        image = cv2.resize(image[:, :], (86, 86),interpolation=cv2.INTER_AREA)
        image = image.reshape(86, 86, 1)
        if not label:
            image -= np.min(image)
            image /= (np.max(image)-np.min(image))
        temp = np.zeros([96, 96, 1])
        temp[10:,10:,:] = image
        image = temp
    elif method == 'gtv':
#         image = cv2.resize(image[:, :], (128, 128),interpolation=cv2.INTER_AREA)
        image = image.reshape(512, 512, 1)
        if not label:
            image -= np.min(image)
            image /= (np.max(image)-np.min(image))
    elif method == 'flair_reg':
        image = cv2.resize(image[:, :], (128, 128),interpolation=cv2.INTER_AREA)
        image = image.reshape(128, 128, 1)
        if not label:
            image -= np.min(image)
            image /= (np.max(image)-np.min(image))
    
    return image
