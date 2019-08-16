import cv2
import nibabel as nib
import numpy as np


def get_extrema(unique_index, mask_extrema):

    d=0
    if len(mask_extrema) == 1:
        min_ind = unique_index[0]-1
        max_ind = unique_index[-1]+1
    else:
        for i in range(1, len(mask_extrema)): 
            d1 = unique_index[mask_extrema[i]-1]-unique_index[mask_extrema[i-1]] 
            if d1>d: 
                min_ind = unique_index[mask_extrema[i-1]]-1
                max_ind = unique_index[mask_extrema[i]-1]+1
                d=d1

    return min_ind, max_ind

def micro_ct_resampling(mask, shape):
 
#     mask = '/mnt/sdb/micro_CT_new_smooth_lung_seg_mean.nii.gz'
#     mask = nib.load(mask).get_data()
     
     
    mask_resampled = np.zeros((shape[0], shape[1], mask.shape[2]))
     
    for z in range(mask.shape[2]): 
        m = cv2.resize(mask[:, :, z], (shape[0], shape[1]), interpolation=cv2.INTER_LINEAR) 
        mask_resampled[:, :, z] = m 
     
#     im2save = nib.Nifti1Image(mask_resampled, affine=np.eye(4))
#     nib.save(im2save, '/mnt/sdb/micro_CT_new_smooth_lung_seg_mean_4.nii.gz')
    return mask_resampled


# def micro_ct_resampling(mask, shape, n_batch):
# 
#     mask_resampled = np.zeros((shape[0], shape[1], int(mask.shape[2]/n_batch)))
#     j = 0
#     for z in range(0, mask.shape[2], n_batch):
#         dimX = shape[0]
#         for b in range(n_batch):
#             m = mask[:, :, z+b]
#             if dimX > 86:
#                 mask_resampled[b*86:(b+1)*86, b*86:(b+1)*86, j] = m
#             else:
#                 mask_resampled[b*86:(b+1)*86, b*86:(b+1)*86, j] = m[:dimX, :dimX]
#             dimX = dimX-86
#         j += 1
# #         m = cv2.resize(mask[:, :, z], (shape[0], shape[1]))#, interpolation=cv2.INTER_NEAREST) 
# #         mask_resampled[:, :, z] = m 
#     
# #     im2save = nib.Nifti1Image(mask_resampled, affine=np.eye(4))
# #     nib.save(im2save, '/mnt/sdb/micro_CT_new_smooth_lung_seg_mean_4.nii.gz')
#     return mask_resampled


def micro_ct_cleaning(mask):
    
    x, y, z = np.where(mask==1)
    
    unique_z = list(set(z))
    unique_y = list(set(y))
    unique_x = list(set(x))
    
    mask_extrema_z = [0]+[i for i in range(1, len(unique_z)) if unique_z[i]-1 != unique_z[i-1]]
    mask_extrema_y = [0]+[i for i in range(1, len(unique_y)) if unique_y[i]-1 != unique_y[i-1]]
    mask_extrema_x = [0]+[i for i in range(1, len(unique_x)) if unique_x[i]-1 != unique_x[i-1]]+[unique_x[-1]]
    
    zmin, zmax = get_extrema(unique_z, mask_extrema_z)
    ymin, ymax = get_extrema(unique_y, mask_extrema_y)
    xmin, xmax = get_extrema(unique_x, mask_extrema_x)
    
    mask_clean = np.zeros((mask.shape))
    mask_clean[xmin:xmax, ymin:ymax, zmin:zmax] = mask[xmin:xmax, ymin:ymax, zmin:zmax]
    
    return mask_clean

def micro_ct_preproc(image):
     
    im = nib.load(image).get_data()
    shape = im.shape
 
    im2 = np.zeros((86, 86, im.shape[2]))                                                                                                                            
 
    for z in range(im.shape[2]):
         
        m = cv2.resize(im[:, :, z], (86, 86),interpolation=cv2.INTER_AREA)
        m = cv2.GaussianBlur(m, (7, 7), 0)
        im2[:,:,z] = m
     
    return im2, shape

# def micro_ct_preproc(image):
#     
#     im = nib.load(image).get_data()
#     shape = im.shape
#     batches = int(np.ceil(shape[0]/86))
#     
#     im2 = np.zeros((86, 86, im.shape[2]*batches))                                                                                                                              
#     j = 0
#     for z in range(im.shape[2]):
#         for b in range(batches):
#             m = np.zeros((86, 86))
#             m1 = im[b*86:(b+1)*86, b*86:(b+1)*86, z]
#             m[:m1.shape[0], :m1.shape[1]] = m1
# #             m = cv2.resize(im[:, :, z], (86, 86),interpolation=cv2.INTER_CUBIC)
#             m = cv2.GaussianBlur(m, (9, 9), 0)
#             im2[:, :, j] = m
#             j += 1
#     
#     return im2, shape, batches