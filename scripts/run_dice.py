import nibabel as nib
import numpy as np
import sys


def dice_calculation(gt, seg):

    gt = nib.load(gt).get_data() 
    seg = nib.load(seg).get_data() 
#     seg = np.squeeze(seg)
    gt = gt.astype('uint16')
    seg = seg.astype('uint16')
    vox_gt = np.sum(gt) 
    vox_seg = np.sum(seg) 
    common = np.sum(gt & seg) 
    dice = (2*common)/(vox_gt+vox_seg) 
    return dice


ref = sys.argv[1]
seg = sys.argv[2]

dice = dice_calculation(ref, seg) 
print(dice)