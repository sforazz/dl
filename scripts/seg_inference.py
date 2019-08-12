from dl.models.unet import mouse_lung_seg
from dl.utils.mouse_segmentation import save_results, preprocessing, postprocessing
from basecore.process.postprocess import binarization, cluster_correction
import nibabel as nib
import numpy as np
import time
from scripts.micro_CT_segmentation import micro_ct_resampling, micro_ct_preproc,\
    micro_ct_cleaning


def seg_inference(list_images, weights, micro_ct=False):

    test_set = []
    n_slices = []
    shapes = []
#     batches = []

    for im in list_images:
        if not micro_ct:
            im = nib.load(im).get_data()
#             n_batch = 1
        else:
            im, shape = micro_ct_preproc(im)
            shapes.append([shape[0], shape[1]])
#             batches.append(n_batch)
        n_slices.append(im.shape[2])
        for s in range(im.shape[2]):
            sl = preprocessing(im[:, :, s])
            test_set.append(sl)
    
    test_set = np.asarray(test_set)
    predictions = []
    model = mouse_lung_seg()
    for i, w in enumerate(weights):
        print('\nSegmentation inference fold {}...\n'.format(i+1))
        model.load_weights(w)
        predictions.append(model.predict(test_set))
        
    predictions = np.asarray(predictions, dtype=np.float32)
    prediction = np.mean(predictions, axis=0)
    # model.load_weights(model_weights)
    # 
    # print('Inference started...')
    # prediction = model.predict(test_set)
    # print('inference ended!')
    
    z = 0
    print('\nBinarizing and saving the results...')
    for i, s in enumerate(n_slices):
        im = prediction[z:z+s, :, :, 0]
        im = postprocessing(im)
        
#         im = binarization(im)
        if micro_ct:
            im = micro_ct_resampling(im, shapes[i])
#         if micro_ct:
#             im = micro_ct_cleaning(im)
        im2correct = save_results(im, list_images[i])
        cluster_correction(im2correct, 0.6, 10000)
        z = z + s


# weights = ['/home/fsforazz/Desktop/PhD_project/fibrosis_project/working_weights_lung_seg/double_feat_per_layer_epoch_10_best.h5']
# images = '/media/fsforazz/extra_HD/micro_CT_converted/list_micro_cts.txt'
save_dir = '/home/fsforazz/Desktop/seg_results_cheng'

# weights = ['/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_jacc_loss_cross_val/double_feat_per_layer_epoch_96_fold_1.h5',
#            '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_jacc_loss_cross_val/double_feat_per_layer_epoch_34_fold_2.h5',
#            '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_jacc_loss_cross_val/double_feat_per_layer_epoch_75_fold_3.h5',
#            '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_jacc_loss_cross_val/double_feat_per_layer_epoch_35_fold_4.h5',
#            '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_jacc_loss_cross_val/double_feat_per_layer_epoch_67_fold_5.h5']
weights = ['/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/original/double_feat_per_layer_cross_ent_fold_1.h5',
           '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/original/double_feat_per_layer_cross_ent_fold_2.h5',
           '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/original/double_feat_per_layer_cross_ent_fold_3.h5',
           '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/original/double_feat_per_layer_cross_ent_fold_4.h5',
           '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/original/double_feat_per_layer_cross_ent_fold_5.h5']

micro_ct = True
 
start = time.perf_counter()
# with open(images, 'r') as f:
#     list_images = [x.strip() for x in f]
list_images = ['/mnt/sdb/results_micro_CT/Cont_Mouse_1li_v1.nii.gz']
seg_inference(list_images, weights, micro_ct=micro_ct)
stop = time.perf_counter()
print('\nAll done!'),
print('Elapsed time: {} seconds'.format(int(stop-start)))
