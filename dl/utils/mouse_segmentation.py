import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
import os


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
    
    basedir, basename = os.path.split(ref)
    out_basename = basename.split('.')[0]+'_lung_seg.nii.gz'
    if save_dir is not None:
        outname = os.path.join(save_dir, out_basename)
    else:
        outname = os.path.join(basedir, out_basename)
    

    ref = nib.load(ref)

    image = im2save[:, 10:, 10:]
    image = image.reshape(-1, 86, 86)
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 0, 1)
    im2save = nib.Nifti1Image(image, ref.affine)
    nib.save(im2save, outname)


def preprocessing(image, label=False):
    
    image = image.astype('float64')
    image = image.reshape(86, 86, 1)
    if not label:
        image -= np.min(image)
        image /= (np.max(image)-np.min(image))
    temp = np.zeros([96, 96, 1])
    temp[10:,10:,:] = image
    image = temp
    
    return image
