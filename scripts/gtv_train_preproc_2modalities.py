import subprocess as sp
import os
import glob
import nibabel as nib
import cv2
import numpy as np


# flair = sorted(glob.glob('/mnt/sdb/Cinderella_CONVERTED/*/*/FLAIR_bc.nii.gz'))
# subs = [['/'.join(f.split('/')[:-1])+'/T1KM_bet_regCT.nii.gz',f] for f in flair if os.path.isfile('/'.join(f.split('/')[:-1])+'/T1KM_bet_regCT.nii.gz')]
# wd = '/mnt/sdb/data_T1KM_to_FLAIR/'
# processed = []
# 
# for t1, f in subs:
#     try:
#         sub = f.split('/')[-3]
#         print('Processing subject {}'.format(sub))
#         t1_bet_mask = f.split('FLAIR')[0]+'T1_bet_mask.nii.gz'
#         f_bet_mask = f.split('FLAIR')[0]+'FLAIR_bet_mask.nii.gz'
#         affine_mat = f.split('FLAIR')[0]+'FLAIR_reg0GenericAffine.mat'
#         cmd = 'antsApplyTransforms -d 3 -i {0} -n NearestNeighbor -r {1} -t [{2},1] -o {3}'.format(
#             t1_bet_mask, f, affine_mat, f_bet_mask)
#         sp.check_output(cmd, shell=True)
#         f_bet = f.split('.')[0]+'_bet.nii.gz'
#         cmd = 'fslmaths {0} -mas {1} {2}'.format(f, f_bet_mask, f_bet)
#         sp.check_output(cmd, shell=True)
#         output = f.split('.')[0]+'_bet_reg'
#         cmd = 'antsRegistrationSyN.sh -d 3 -f {0} -m {1} -o {2} -t r -n 2'.format(t1,f_bet,output)
#         sp.check_output(cmd, shell=True)
# 
#         processed.append(f)
#     except:
#         print('{} failed!!!'.format({f}))
#         continue


flair = sorted(glob.glob('/mnt/sdb/Cinderella_CONVERTED/gbm/*/*/FLAIR_regCT.nii.gz'))
subs = [['/'.join(f.split('/')[:-1])+'/T1KM_bet_regCT.nii.gz', f, '/'.join(f.split('/')[:-1])+'/gtv.nii.gz']
        for f in flair if os.path.isfile('/'.join(f.split('/')[:-1])+'/T1KM_bet_regCT.nii.gz')]
wd = '/mnt/sdb/data_T1KM_to_FLAIR/'
processed = []

for t1, f, gtv in subs:
    try:
        sub = f.split('/')[-3]
        print('Processing subject {}'.format(sub))
        ref = nib.load(t1)
        t1_data = ref.get_data()
        flair_data = nib.load(f).get_data()
        gtv_data = nib.load(gtv).get_data()
        t1_c = t1.split('.')[0]+'_cropped.nii.gz'
        flair_c = f.split('.')[0]+'_cropped.nii.gz'
        gtv_c = gtv.split('.')[0]+'_cropped.nii.gz'
        x, y, z = np.where(t1_data!=0)
        t1_cropped = t1_data[x.min():x.max(),y.min():y.max(),z.min():z.max()]
        flair_cropped = flair_data[x.min():x.max(),y.min():y.max(),z.min():z.max()]
        gtv_cropped = gtv_data[x.min():x.max(),y.min():y.max(),z.min():z.max()]
        im2save = nib.Nifti1Image(t1_cropped, affine=ref.affine)
        nib.save(im2save, t1_c)
        im2save = nib.Nifti1Image(flair_cropped, affine=ref.affine)
        nib.save(im2save, flair_c)
        im2save = nib.Nifti1Image(gtv_cropped, affine=ref.affine)
        nib.save(im2save, gtv_c)
        output = t1_c.split('.')[0]+'_bet.nii.gz'
        cmd = 'hd-bet -i {0} -o {1} -device 0 -mode accurate -tta 1 -pp 1'.format(t1_c, output)
        sp.check_output(cmd, shell=True)
        mask = t1.split('.')[0]+'_cropped_bet_mask.nii.gz'
        f_bet = flair_c.split('.')[0]+'_bet.nii.gz'
        cmd = 'fslmaths {0} -mas {1} {2}'.format(flair_c, mask, f_bet)
        sp.check_output(cmd, shell=True)

        processed.append(f)
    except:
        print('{} failed!!!'.format({f}))
        continue