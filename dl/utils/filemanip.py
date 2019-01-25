from skimage.io import imread
import os
import nibabel as nib


def get_png(image_path, labels=False):
    
    if labels:
        path, filename = os.path.split(image_path)
        img_number = filename.split('.')[0].split('_')[1]
        mask_path = os.path.join(path, 'Mask_{}.png'.format(img_number))
        img = imread(mask_path)
    else:
        img = imread(image_path)
    
    return(img)


def get_nifti(image_path, labels=False):
    
    if labels:
        path, filename = os.path.split(image_path)
        img_number = filename.split('.')[0].split('_')[1]
        slice_number = filename.split('.')[0].split('_')[2]
        mask_path = os.path.join(path, 'Mask_{0}_{1}.nii.gz'.format(img_number, slice_number))
        img = nib.load(mask_path).get_data()
    else:
        img = nib.load(image_path).get_data()
    
    return(img)
