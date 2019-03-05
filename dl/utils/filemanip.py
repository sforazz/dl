from skimage.io import imread
import os
import nibabel as nib
import numpy as np


def get_png(image_path, labels=False):
    
    if labels:
        path, filename = os.path.split(image_path)
        img_number = filename.split('.')[0].split('_')[1]
        mask_path = os.path.join(path, 'Mask_{}.png'.format(img_number))
        img = imread(mask_path)
    else:
        img = imread(image_path)
    
    return(img)


def get_nifti(image_path, labels=False, method='same_folder'):
    
    if labels:
        path, filename = os.path.split(image_path)
        if method=='same_folder':
            mask_name = filename.split('Raw_data_for_')[-1]
            mask_path = os.path.join(path, mask_name)
            img = nib.load(mask_path).get_data()
        else:
            img_number = filename.split('.')[0].split('_')[1]
            slice_number = filename.split('.')[0].split('_')[2]
            mask_path = os.path.join(path, 'Mask_{0}_{1}.nii.gz'.format(img_number, slice_number))
            img = nib.load(mask_path).get_data()
    else:
        img = nib.load(image_path).get_data()
    
    return(img)


def split_nifti(image):
    
    path, filename = os.path.split(image)
    out_basename = filename.split('.')[0]
    img = nib.load(image).get_data()
    outfiles = []
    for i in range(img.shape[2]):
        outname = out_basename+'_vol_{}.nii.gz'.format(str(i).zfill(4))
        im2save = nib.Nifti1Image(img[:, :, i], affine=np.eye(4))
        nib.save(im2save, os.path.join(path, outname))
        outfiles.append(os.path.join(path, outname))
    
    return outfiles
