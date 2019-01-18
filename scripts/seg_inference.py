from models.unet import mouse_lung_seg, UNet
from utils.mouse_segmentation import data_preparation, save_results
from utils.image_transform import binarization
import nibabel as nib
import os
from dask import base


def segmentation_inference(image, model, model_weights, save_dir=None, im_dir=''):
    
    basename = os.path.basename(image).split('.')[0]
    image = os.path.join(im_dir, basename)+'.nii.gz'
    image_dir = os.path.dirname(image)
    out_basename = basename+'_lung_seg.nii.gz'
    if save_dir is not None:
        outname = os.path.join(save_dir, out_basename)
    else:
        outname = os.path.join(image_dir, out_basename)
    ref = nib.load(image)

    data = data_preparation([image])

#     model = mouse_lung_seg()
    model.load_weights(model_weights)

    prediction = model.predict(data)
    prediction_bin = binarization(prediction)
    
    save_results(prediction_bin, ref, outname)

model_weights = '/home/fsforazz/Desktop/git/deep_learning/scripts/lung_bs=60_spe=172_e=8+2_loss=bin_crossEntropy_metrics=jacc_dist.h5'
images = '/home/fsforazz/Desktop/mouse_nifti/images_for_test.txt'
save_dir = '/home/fsforazz/Desktop/mouse_segmentation_results'

with open(images, 'r') as f:
    list_images = [x.strip() for x in f]

model = mouse_lung_seg()

for image in list_images:
    segmentation_inference(image, model, model_weights, save_dir=save_dir, im_dir='/home/fsforazz/Desktop/mouse_nifti/')