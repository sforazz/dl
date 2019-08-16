from dl.models.unet import mouse_lung_seg
from dl.utils.mouse_segmentation import save_results, preprocessing, postprocessing
from basecore.process.postprocess import binarization, cluster_correction
import nibabel as nib
import numpy as np
import time
from dl.generators import load_data_2D
from dl.utils.data_utils import save_prediction_2D


def seg_inference(list_images, weights):

    test_set = None
#     test_set = []
    n_slices = []
    patches = []
    dicts = []

    for im in list_images:
        im = nib.load(im).get_data()
        n_slices.append(im.shape[2])
        for s in range(im.shape[2]):
            sl, result_dict, pt = load_data_2D('', '', array=im[:, :, s], extract_edges=False, prediction=True)
            patches.append(pt)
#             sl = preprocessing(im[:, :, s], method='human')
#             test_set.append(sl)
            if test_set is not None:
                    test_set = np.concatenate([test_set, sl], axis=0)
            else:
                    test_set = sl
        dicts.append(result_dict)
        patches.append(pt)

    test_set = np.asarray(test_set)
#     test_set = test_set.reshape(320*36,96,96,1)
    predictions = []
    model = mouse_lung_seg()
    for i, w in enumerate(weights):
        print('\nSegmentation inference fold {}...\n'.format(i+1))
        model.load_weights(w)
        predictions.append(model.predict(test_set))
        
    predictions = np.asarray(predictions, dtype=np.float32)
    prediction = predictions[4, :]
#     prediction = np.mean(predictions, axis=0)
    
    z = 0
    print('\nBinarizing and saving the results...')
    for i, s in enumerate(n_slices):
        final_image = None
#         im = prediction[z:z+s, :, :, :]
        im = prediction[z:z+(s*patches[i]), :, :, :]
        for sl in range(0, im.shape[0], patches[i]):
            recon_im = save_prediction_2D(im[sl:sl+patches[i], :, :, :], dict_val=dicts[i], binarize=False)
            recon_im = recon_im.reshape(recon_im.shape[0], recon_im.shape[1], 1)
            if final_image is not None:
                final_image = np.concatenate([final_image, recon_im], axis=-1)
            else:
                final_image = recon_im
#         im = postprocessing(im, method='human')
# #         
#         im = binarization(im)
#         final_image = binarization(final_image)
        im2correct = save_results(final_image, list_images[i])
        cluster_correction(im2correct, 0.2, 10000)
        z = z+(s*patches[i])
#         save_results(im, list_images[i])
#         z = z+s


# weights = ['/home/fsforazz/Desktop/PhD_project/fibrosis_project/working_weights_lung_seg/double_feat_per_layer_epoch_10_best.h5']
images = '/media/fsforazz/extra_HD/micro_CT_converted/list_micro_cts.txt'
save_dir = '/home/fsforazz/Desktop/seg_results_cheng'

# weights = ['/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_jacc_loss_cross_val/double_feat_per_layer_epoch_96_fold_1.h5',
#            '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_jacc_loss_cross_val/double_feat_per_layer_epoch_34_fold_2.h5',
#            '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_jacc_loss_cross_val/double_feat_per_layer_epoch_75_fold_3.h5',
#            '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_jacc_loss_cross_val/double_feat_per_layer_epoch_35_fold_4.h5',
#            '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_jacc_loss_cross_val/double_feat_per_layer_epoch_67_fold_5.h5']
weights = ['/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/double_feat_per_layer_cross_ent_fold_1.h5',
           '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/double_feat_per_layer_cross_ent_fold_2.h5',
           '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/double_feat_per_layer_cross_ent_fold_3.h5',
           '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/double_feat_per_layer_cross_ent_fold_4.h5',
           '/home/fsforazz/Desktop/PhD_project/fibrosis_project/weights_bin_crossEnt_CV_whole_ts/double_feat_per_layer_cross_ent_fold_5.h5']
 
start = time.perf_counter()
# with open(images, 'r') as f:
#     list_images = [x.strip() for x in f]
list_images = ['/mnt/sdb/results_micro_CT/Cont_mouse_1li+1re_v1.nii.gz']
seg_inference(list_images, weights)
stop = time.perf_counter()
print('\nAll done!'),
print('Elapsed time: {} seconds'.format(int(stop-start)))
