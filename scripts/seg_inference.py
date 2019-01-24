from models.unet import mouse_lung_seg
from utils.mouse_segmentation import save_results, preprocessing
from utils.image_transform import binarization
import nibabel as nib
import numpy as np


model_weights = '/home/fsforazz/git/deep_learning/scripts/double_feat_per_layer_epoch_10.h5'
images = '/home/fsforazz/Desktop/mouse_nifti/images_for_test2.txt'
save_dir = '/home/fsforazz/Desktop/mouse_segmentation_results'

with open(images, 'r') as f:
    list_images = [x.strip() for x in f]

test_set = []
n_slices = []
for im in list_images:
    im = nib.load(im).get_data()
    n_slices.append(im.shape[2])
    for s in range(im.shape[2]):
        sl = preprocessing(im[:, :, s])
        test_set.append(sl)

test_set = np.asarray(test_set)

model = mouse_lung_seg()
model.load_weights(model_weights)

prediction = model.predict(test_set)

z = 0
for i, s in enumerate(n_slices):
    im = prediction[z:z+s, :, :, 0]
    im = binarization(im)
    save_results(im, list_images[i], save_dir)
    z = z + s

print('Done!')
