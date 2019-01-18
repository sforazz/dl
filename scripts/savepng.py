import random
import numpy as np
import nibabel as nib
import glob
import matplotlib.pyplot as plot
import multiprocessing
import os


def save_png(args):
    
    images = args[0] 
    el = args[1] 
    n_slice = args[2] 
    val_list = args[3]
    data_dir = args[4]
    print('Processing {}'.format(images[el]))
    im = nib.load(images[el]).get_data() 
    for s in range(im.shape[2]): 
        plot.imshow(im[:,:,s], cmap='gray') 
        if el not in val_list: 
            plot.savefig(data_dir+'/training/Mask_{}.png'.format(str(n_slice[s]).zfill(8))) 
        else: 
            plot.savefig(data_dir+'/validation/Mask_{}.png'.format(str(n_slice[s]).zfill(8))) 
        plot.close()

data_dir = '/home/fsforazz/Desktop/mouse_nifti'
images = sorted(glob.glob(data_dir+'/Mouse_0*.nii.gz'))
masks = sorted(glob.glob(data_dir+'/Mask_0*.nii.gz'))

try:
    train_indexs = []
    with open(os.path.join(data_dir, 'indexs_for_training.txt'), 'r') as f:
        for ind in f:
            train_indexs.append(int(ind.strip()))
except FileNotFoundError:
    test_indexs = random.sample(range(len(images)), 170)
    train_indexs = [x for x in range(len(images)) if x not in test_indexs]
    with open(data_dir+'/indexs_for_training.txt', 'w') as f:
        for el in train_indexs:
            f.write(str(el)+'\n')
    with open(data_dir+'/images_for_test.txt', 'w') as f: 
        for el in test_indexs: 
            f.write(images[el]+'\n')

try:
    val_indexs = []
    with open(os.path.join(data_dir, 'indexs_for_validation.txt'), 'r') as f:
        for ind in f:
            val_indexs.append(int(ind.strip()))
except FileNotFoundError:
    val_indexs = random.sample(train_indexs, 300)
    with open(data_dir+'/indexs_for_validation.txt', 'w') as f:
        for el in val_indexs:
            f.write(str(el)+'\n')

n_slices_tot = []
for el in train_indexs: 
    im = nib.load(masks[el]).get_data() 
    n_slices_tot.append(im.shape[2])

slice_indexs = []
z=0 
for n_slice in n_slices_tot: 
    slice_indexs.append(np.arange(z,z+n_slice).tolist())
    z = z+n_slice 

print('Starting multiprocessing...')
p = multiprocessing.Pool(processes=3)
p.map(save_png, zip([masks]*1500, train_indexs, slice_indexs, [val_indexs]*1500, [data_dir]*1500)) 
