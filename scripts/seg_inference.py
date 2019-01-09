from models.unet import mouse_lung_seg
from utils.mouse_segmentation import data_preparation, save_results
from utils.image_transform import binarization
import nibabel as nib
import os


def segmentation_inference(image, model_weights, save_dir=None):
    
    basename = os.path.basename(image).split('.')[0]
    image_dir = os.path.dirname(image)
    out_basename = basename+'_lung_seg.nii.gz'
    if save_dir is not None:
        outname = os.path.join(save_dir, out_basename)
    else:
        outname = os.path.join(image_dir, out_basename)
    ref = nib.load(image)

    data = data_preparation([image])

    model = mouse_lung_seg()
    model.load_weights(model_weights)

    prediction = model.predict(data)
    prediction_bin = binarization(prediction)
    
    save_results(prediction_bin, ref, outname)

model_weights = '/home/fsforazz/git/deep_learning/scripts/lung_bs=60_spe=172_e=10_loss=bin_crossEntropy_metrics=jacc_dist.h5'
image = '/home/fsforazz/Desktop/mouse_nifti/Mouse_01479.nii.gz'
save_dir = '/home/fsforazz/Desktop/mouse_segmentation_results'

segmentation_inference(image, model_weights, save_dir=save_dir)