import glob
import subprocess as sp


diffImages = sorted(glob.glob('/home/fsforazz/Desktop/mouse_diff_images/Mask_0*_diff.nii.gz'))
regDir = '/home/fsforazz/Desktop/mouse_reg_new_template/'

for im in diffImages:
    im_number = im.split('/')[-1].split('_')[1]
    outname = im.split('/')[-1].split('.')[0]+'_reg.nii.gz'
    try:
        cmd = ('antsApplyTransforms -d 3 -i {0} -t {1}Mouse_{2}_reg_1Warp.nii.gz -t {1}Mouse_{2}_reg_0GenericAffine.mat '
               '-n NearestNeighbor -r /home/fsforazz/Desktop/mouse_template/image4template/mouse_templatetemplate0.nii.gz'
               ' -o {1}aligned_diff/{3}'.format(im, regDir, im_number, outname))
        sp.check_output(cmd, shell=True)
    except:
        continue

print('Done!')

