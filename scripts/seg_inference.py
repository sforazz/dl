from dl.models.unet import mouse_lung_seg
from dl.utils.mouse_segmentation import save_results, preprocessing
from core.process.postprocess import binarization
import nibabel as nib
import numpy as np
import time


# model_weights = ('/home/fsforazz/Desktop/PhD_project/fibrosis_project/working_weights_lung_seg/double_feat_per_layer_epoch_10_best.h5')
images = '/mnt/sdb/mouse_data_prep_new/Test_files.txt'
save_dir = '/home/fsforazz/Desktop/seg_results_cheng'

start = time.perf_counter()
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
weights = ['/home/fsforazz/git/deep_learning/scripts/double_feat_per_layer_epoch_96_fold_1.h5',
           '/home/fsforazz/git/deep_learning/scripts/double_feat_per_layer_epoch_34_fold_2.h5',
           '/home/fsforazz/git/deep_learning/scripts/double_feat_per_layer_epoch_75_fold_3.h5']
predictions = []
model = mouse_lung_seg()
for i, w in enumerate(weights):
    print('\nSegmentation inference fold {}...\n'.format(i+1))
    model.load_weights(w)
    predictions.append(model.predict(test_set))
    
predictions = np.asarray(predictions)
prediction = np.mean(predictions, axis=2)
# model.load_weights(model_weights)
# 
# print('Inference started...')
# prediction = model.predict(test_set)
# print('inference ended!')

z = 0
print('\nBinarizing and saving the results...')
for i, s in enumerate(n_slices):
    im = prediction[z:z+s, :, :, 0]
    im = binarization(im)
    save_results(im, list_images[i])
    z = z + s

stop = time.perf_counter()
print('\nAll done!'),
print('Elapsed time: {} seconds'.format(int(stop-start)))
