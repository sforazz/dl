import numpy as np
import os
from dl.generators import load_data_3D


dset = '/mnt/sdb/BRATS2015_Training/HGG/'
extract_edges = False
subjects_per_batch = 110
init = 0
subject_number = 220
channels = ['OT']
merge = True
save_single = True
z = 0

if not extract_edges:
    for n, i in enumerate(range(0, subject_number, subjects_per_batch)):
        all_tensor = None
        for c in channels:
            img_type = '*/*{}.*/*.nii.gz'.format(c)
            X_full_train, _ = load_data_3D(
                dset, img_type, extract_edges=extract_edges, bs=i+subjects_per_batch, init=i,
                normalize=False, binarize=True)
            if merge:
                if all_tensor is None:
                    all_tensor = X_full_train
                else:
                    all_tensor = np.concatenate([all_tensor, X_full_train], axis=-1)
            else:
                if save_single:
                    for patch in range(X_full_train.shape[0]):
                        np.save(os.path.join(dset, 'training_{0}_patch{1}.npy'.format(c, str(z).zfill(7))), X_full_train[patch, :, :, :, :])
                        z += 1
                else:
                    np.save(os.path.join(dset, 'training_{0}_part{1}.npy'.format(c, n)), X_full_train)
        if merge:
            if save_single:
                for patch in range(all_tensor.shape[0]):
                    np.save(os.path.join(dset, 'single_patch_data', 'training_{0}_patch{1}.npy'
                                         .format('+'.join(channels), str(z).zfill(7))), all_tensor[patch, :, :, :, :])
                    z += 1
            else:
                np.save(os.path.join(dset, 'training_{0}_part{1}.npy'.format('+'.join(channels), n)), all_tensor)
else:
    raise NotImplementedError('Edge calculation is not supported yet.')

print('Done!')
