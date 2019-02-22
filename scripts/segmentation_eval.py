import nibabel as nib
import numpy as np
import glob
import matplotlib.pyplot as plot
from core.process.postprocess import eucl_max
from core.utils.plot import violin_box_plot


def dice_calculation(gt, seg):

    gt = nib.load(gt).get_data() 
    seg = nib.load(seg).get_data() 
    seg = np.squeeze(seg)
    gt = gt.astype('uint16')
    seg = seg.astype('uint16')
    vox_gt = np.sum(gt) 
    vox_seg = np.sum(seg) 
    common = np.sum(gt & seg) 
    dice = (2*common)/(vox_gt+vox_seg) 
    return dice


def save_difference(gt, seg):
    
    outname = gt.split('/')[-1].split('.')[0]+'_diff.nii.gz'
    gt1 = nib.load(gt).get_data() 
    seg = nib.load(seg).get_data() 
    seg = np.squeeze(seg)
    diff = gt1 - seg
    im2save = nib.Nifti1Image(diff, affine=nib.load(gt).affine)
    nib.save(im2save, '/home/fsforazz/Desktop/mouse_diff_images/{}'.format(outname))


def outliers_modified_z_score(ys):

    ys = np.asarray(ys)
    threshold = 3.5

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return np.where(np.abs(modified_z_scores) > threshold)


refs = sorted(glob.glob('/home/fsforazz/Desktop/mouse_nifti/Mask_0*.nii.gz'))
segs = sorted(glob.glob('/home/fsforazz/Desktop/mouse_segmentation_results/*.nii.gz'))
all_dices = []
all_hd = []
all_hd95 = []

for seg in segs: 
    seg_num = seg.split('/')[-1].split('_')[1] 
    ref = [x for x in refs if seg_num in x][0] 
    dice = dice_calculation(ref, seg)
    save_difference(ref, seg)
    hd_rs = eucl_max(ref, seg)
    hd_sr = eucl_max(seg, ref)
    hd95_rs = eucl_max(ref, seg, percentile=99)
    hd95_sr = eucl_max(seg, ref, percentile=99)
    all_dices.append(dice)
    all_hd.append(np.max([hd_rs, hd_sr]))
    all_hd95.append(np.max([hd95_rs, hd95_sr]))

outliers = outliers_modified_z_score(all_dices)
print('Mean DSC: {0} \nMedian DSC: {4} \nStd: {1} \nMax DSC: {2}\nMin DSC: {3}'
      .format(np.mean(all_dices), np.std(all_dices), np.max(all_dices), np.min(all_dices),
              np.median(all_dices)))
for ol in outliers[0]:
    print('Outlier: {0}, DSC: {1}'.format(segs[ol], all_dices[ol]))

violin_box_plot(all_dices)
plot.violinplot(all_hd)

print('Done!')
