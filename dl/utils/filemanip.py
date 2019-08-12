from skimage.io import imread
import os
import nibabel as nib
import numpy as np
from random import sample


def get_png(image_path, labels=False):
    
    if labels:
        path, filename = os.path.split(image_path)
        img_number = filename.split('.')[0].split('_')[1]
        mask_path = os.path.join(path, 'Mask_{}.png'.format(img_number))
        img = imread(mask_path)
    else:
        img = imread(image_path)
    
    return(img)


def get_nifti(image_path, labels=False, method='mouse_fibrosis'):
    
    if labels:
        path, filename = os.path.split(image_path)
        if method=='mouse_fibrosis':
            mask_name = filename.split('Raw_data_for_')[-1]
            mask_path = os.path.join(path, mask_name)
            img = nib.load(mask_path).get_data()
        elif method=='gtv':
            mask_vol = filename.split('CT_vol_')[-1]
            mask_name = 'gtv_vol_'+mask_vol
            mask_path = os.path.join(path, mask_name)
            img = nib.load(mask_path).get_data()
        elif method == 'micro_ct':
            mask_vol = filename.split('_vol_')
            mask_name = mask_vol[0]+'_mask_vol_'+mask_vol[1]
            mask_path = os.path.join(path, mask_name)
            img = nib.load(mask_path).get_data()
        elif method == 'flair_reg':
            mask_vol = filename.split('_vol_')[-1]
            mask_name = 'MR_FLAIR_reg_bet_vol_'+mask_vol
            mask_path = os.path.join(path, mask_name)
            img = nib.load(mask_path).get_data()
        elif method == 'human':
            mask_vol = filename.split('_vol')
            mask_name = mask_vol[0]+'_lungs_vol'+mask_vol[1]
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


def data_split(data_dir, save_test_set=True, split=True, train_len=0, test_len=169):
    
    data = []
    masks = []
    train_files = []
    validation_files = []

    if split:
        for root, _, files in os.walk(data_dir): 
            for name in files: 
                if name.endswith('.nii.gz') and 'Raw_data' in name and 'lung_seg' not in name: 
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
        test_set_indxs = sample(range(len(data)), test_len)
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
