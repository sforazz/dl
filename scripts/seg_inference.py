from models.unet import mouse_lung_seg_less_feat, mouse_lung_seg
from utils.mouse_segmentation import data_preparation, save_results
from utils.image_transform import binarization
import nibabel as nib
import os


def segmentation_inference(image, model, save_dir=None, im_dir=''):
    
    basename = os.path.basename(image).split('.')[0]
    image = os.path.join(im_dir, basename)+'.nii.gz'
    image_dir = os.path.dirname(image)
    out_basename = basename+'_lung_seg.nii.gz'
    if save_dir is not None:
        outname = os.path.join(save_dir, out_basename)
    else:
        outname = os.path.join(image_dir, out_basename)
    ref = nib.load(image)

    data = data_preparation([image], preproc='zero-one')

    prediction = model.predict(data)
    prediction_bin = binarization(prediction)
    
    save_results(prediction_bin, ref, outname)


def save_nii(predictions, slices, ref):

    for i, s in enumerate(slices):
        image = predictions[i*s:(i+1)*s]
        image = binarization(image)
        save_results(image, ref, 'Image_{}.nii.gz'.format(str(i+1).zfill(4)))


def split_nii(image):

    basename = os.path.basename(image).split('.')[0]
    dir_name = os.path.dirname(image)
    out_dir = os.path.join(dir_name, 'test_dir')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    ref = nib.load(image)
    image = ref.get_data()

    splitted = []
    for s in range(image.shape[2]): 
        outname =  out_dir+'/{0}_slice_{1}.nii.gz'.format(basename, str(s).zfill(4))
        im2save = nib.Nifti1Image(image[:, :, s], affine=ref.affine)
        nib.save(im2save, outname)
        splitted.append(outname)
    
    return splitted

model_weights = '/home/fsforazz/git/deep_learning/scripts/test_weights_epoch_17.h5'
images = '/home/fsforazz/Desktop/mouse_nifti/images_for_test2.txt'
save_dir = '/home/fsforazz/Desktop/mouse_segmentation_results'

with open(images, 'r') as f:
    list_images = [x.strip() for x in f]

model = mouse_lung_seg_less_feat()
model.load_weights(model_weights)

for image in list_images:
    segmentation_inference(image, model, save_dir=save_dir, im_dir='/home/fsforazz/Desktop/mouse_nifti/')


print('Done!')
