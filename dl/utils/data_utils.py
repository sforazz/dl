import nibabel as nib
import os
import numpy as np
import cv2
from skimage.transform import resize
import glob
from dl.utils.utilities import sobel_3D, sobel_2D


def normalize_array_max(array):
    max_value = max(array.flatten())
    if max_value > 0:
        array = array / max_value
        array = (array - 0.5)*2
    return array, max_value


def inverse_normalize_array_max(array, max_val):

    array = (array + 1)/2
    array = array * max_val
    return array


def get_nb_patch(img_dim, patch_size, image_data_format):

    assert image_data_format in ["channels_first", "channels_last"], "Bad image_data_format"

    if image_data_format == "channels_first":
        assert img_dim[1] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[2] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] // patch_size[1])
        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])

    elif image_data_format == "channels_last":
        assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
        img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])

    return nb_patch, img_dim_disc


def get_nb_patch_3D(img_dim, patch_size, image_data_format):

    assert image_data_format in ["channels_first", "channels_last"], "Bad image_data_format"

    if image_data_format == "channels_first":
        assert img_dim[1] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[2] % patch_size[1] == 0, "patch_size does not divide width"
        assert img_dim[3] % patch_size[2] == 0, "patch_size does not divide depth"
        nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] // patch_size[1]) * (img_dim[3] // patch_size[2])
        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1], patch_size[2])

    elif image_data_format == "channels_last":
        assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
        assert img_dim[2] % patch_size[2] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1]) * (img_dim[2] // patch_size[2])
        img_dim_disc = (patch_size[0], patch_size[1], patch_size[2], img_dim[-1])

    return nb_patch, img_dim_disc


def extract_patches(X, image_data_format, patch_size):

    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        X = X.transpose(0,2,3,1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)

    return list_X


def extract_patches_3D(X, image_data_format, patch_size):

    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        X = X.transpose(0,2,3,1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]
    list_sli_idx = [(i * patch_size[2], (i + 1) * patch_size[2]) for i in range(X.shape[3] // patch_size[2])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            for sli_idx in list_sli_idx:
                list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], sli_idx[0]:sli_idx[1], :])

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)

    return list_X


def load_data(data_dir, data_type, image_data_format, img_width=256, img_height=256, extract_edges=False):

        # Get all .h5 files containing training images
    facade_photos_h5 = sorted(os.listdir(os.path.join(data_dir, data_type+'A')))
    facade_labels_h5 = sorted(os.listdir(os.path.join(data_dir, data_type+'B')))
#     facade_labels_h5 = [f for f in os.listdir(os.path.join(data_dir_path, 'facades')) if '.h5' in f]

    final_facade_photos = None
    final_facade_labels = None
    final_photo_edges = None
    final_label_edges = None
    
    for index in range(len(facade_photos_h5)):
        facade_photos_path = os.path.join(data_dir, data_type+'A/') + facade_photos_h5[index]
        facade_labels_path = os.path.join(data_dir, data_type+'B/') + facade_labels_h5[index]
#         facade_labels_path = data_dir_path + '/facades/' + facade_labels_h5[index]
        facade_photos_orig = nib.load(facade_photos_path).get_data()
        facade_photos_orig = cv2.resize(facade_photos_orig, (img_width, img_height), interpolation=cv2.INTER_AREA)
        facade_photos, _ = normalize_array_max(facade_photos_orig)
        facade_labels_orig = nib.load(facade_labels_path).get_data()
        facade_labels_orig = cv2.resize(facade_labels_orig, (img_width, img_height), interpolation=cv2.INTER_AREA)
        facade_labels, _ = normalize_array_max(facade_labels_orig)
        # Resize and normalize images
#         num_photos = facade_photos['data'].shape[0]
#         num_labels = facade_labels['data'].shape[0]
    
#         all_facades_photos = np.array(facade_photos['data'], dtype=np.float32)
        all_facades_photos = facade_photos.reshape((1, img_width, img_height, 1))
        
#         all_facades_labels = np.array(facade_labels['data'], dtype=np.float32)
        all_facades_labels = facade_labels.reshape((1, img_width, img_height, 1))
        
        if final_facade_photos is not None and final_facade_labels is not None:
                    final_facade_photos = np.concatenate([final_facade_photos, all_facades_photos], axis=0)
                    final_facade_labels = np.concatenate([final_facade_labels, all_facades_labels], axis=0)
        else:
                    final_facade_photos = all_facades_photos
                    final_facade_labels = all_facades_labels
        if extract_edges:
            photo_edges = sobel_2D(facade_photos_orig)
            label_edges = sobel_2D(facade_labels_orig)
#             photo_edges, _ = normalize_array_max(facade_photos_edge)
#             label_edges, _ = normalize_array_max(facade_labels_edge)
            all_photo_edges = photo_edges.reshape((-1, img_width, img_height, 2))
            all_label_edges = label_edges.reshape((-1, img_width, img_height, 2))
            if final_photo_edges is not None and final_label_edges is not None:
                final_photo_edges = np.concatenate([final_photo_edges, all_photo_edges], axis=0)
                final_label_edges = np.concatenate([final_label_edges, all_label_edges], axis=0)
            else:
                final_photo_edges = all_photo_edges
                final_label_edges = all_label_edges
    
    if extract_edges:
        return final_facade_photos, final_facade_labels, final_photo_edges, final_label_edges
    else:
        return final_facade_photos, final_facade_labels


def load_data_3D(data_dir, data_type, bias_corr=False, img_width=128, img_height=128, img_depth=128, mb=[2, 2, 2], bs=None,
                 init=None, extract_edges=True):

        # Get all .h5 files containing training images
    if bias_corr:
        data_dir = os.path.join(data_dir, 'bias_corr')
#     facade_photos_h5 = sorted(os.listdir(os.path.join(data_dir, data_type+'A')))
#     facade_labels_h5 = sorted(os.listdir(os.path.join(data_dir, data_type+'B')))
    if bs is not None and init is not None:
        facade_photos_h5 = sorted(glob.glob(os.path.join(data_dir, '*/*T1c.*/*.nii.gz')))[init:bs]
        facade_labels_h5 = sorted(glob.glob(os.path.join(data_dir, '*/*Flair.*/*.nii.gz')))[init:bs]
    elif bs is not None and init is None:
        idx = np.random.choice(len(facade_photos_h5), bs, replace=False)
        facade_photos_h5 = [facade_photos_h5[x] for x in idx]
        facade_labels_h5 = [facade_labels_h5[x] for x in idx]
    else:
        facade_photos_h5 = sorted(os.listdir(os.path.join(data_dir, data_type+'A')))
        facade_labels_h5 = sorted(os.listdir(os.path.join(data_dir, data_type+'B')))
#     facade_labels_h5 = [f for f in os.listdir(os.path.join(data_dir_path, 'facades')) if '.h5' in f]
    dx = 240
    dy = 240
    dz = 155

    final_facade_photos = None
    final_facade_labels = None
    final_photo_edges = None
    final_label_edges = None
    
    diffX = dx - img_width
    diffY = dy - img_height
    diffZ = dz - img_depth

    while True:
        if diffX % (mb[0]-1) != 0:
            diffX += 1
            dx += 1
        elif diffY % (mb[1]-1) != 0:
            diffY += 1
            dy += 1
        elif diffZ % (mb[2]-1) != 0:
            diffZ += 1
            dz += 1
        else:
            break

    overlapX = diffX//(mb[0]-1)
    overlapY = diffY//(mb[1]-1)
    overlapZ = diffZ//(mb[2]-1)
    indX = [[x, x+img_width] for x in np.arange(0, dx, overlapX) if x+img_width<=dx]
    indY = [[x, x+img_height] for x in np.arange(0, dy, overlapY) if x+img_height<=dy]
    indZ = [[x, x+img_depth] for x in np.arange(0, dz, overlapZ) if x+img_depth<=dz]

    results_dict = {}
    for index in range(len(facade_photos_h5)):
        results_dict[index] = {}
        facade_photos_path = facade_photos_h5[index]
        facade_labels_path = facade_labels_h5[index]
#         facade_labels_path = data_dir_path + '/facades/' + facade_labels_h5[index]
        facade_photos_orig = nib.load(facade_photos_path).get_data()
        if facade_photos_orig.shape != (240,240,155):
            raise Exception('different size')
#         facade_photos_orig = resize(facade_photos_orig, (dx, dy, dz), order=3, mode='edge', cval=0,
#                                     anti_aliasing=False)

        facade_labels_orig = nib.load(facade_labels_path).get_data()
#         facade_labels_orig = resize(facade_labels_orig, (dx, dy, dz), order=3, mode='edge', cval=0,
#                                     anti_aliasing=False)
        
        facades_photo = [facade_photos_orig[i[0]:i[1], j[0]:j[1], z[0]:z[1]] for z in indZ for j in indY for i in indX]
        facades_label = [facade_labels_orig[i[0]:i[1], j[0]:j[1], z[0]:z[1]] for z in indZ for j in indY for i in indX]
        

        facade_photos, max_photos = normalize_array_max(np.asarray(facades_photo, dtype=np.float16))
        facade_labels, max_labels = normalize_array_max(np.asarray(facades_label, dtype=np.float16))
        

        all_facades_photos = facade_photos.reshape((-1, img_width, img_height, img_depth, 1))
        all_facades_labels = facade_labels.reshape((-1, img_width, img_height, img_depth, 1))

        results_dict[index]['max_photos'] = max_photos
        results_dict[index]['max_labels'] = max_labels
        results_dict[index]['indexes'] = [indX, indY, indZ]
        results_dict[index]['im_size'] = [dx, dy, dz]

        if final_facade_photos is not None and final_facade_labels is not None:
                    final_facade_photos = np.concatenate([final_facade_photos, all_facades_photos], axis=0)
                    final_facade_labels = np.concatenate([final_facade_labels, all_facades_labels], axis=0)
        else:
                    final_facade_photos = all_facades_photos
                    final_facade_labels = all_facades_labels

        if extract_edges:
            facade_photos_edge = sobel_3D(facade_photos_orig)
            facade_labels_edge = sobel_3D(facade_labels_orig)
            facades_label_edge = [facade_labels_edge[i[0]:i[1], j[0]:j[1], z[0]:z[1]] for z in indZ for j in indY for i in indX]
            facades_photo_edge = [facade_photos_edge[i[0]:i[1], j[0]:j[1], z[0]:z[1]] for z in indZ for j in indY for i in indX]
            photo_edges = np.asarray(facades_photo_edge, dtype=np.float16)
            label_edges = np.asarray(facades_label_edge, dtype=np.float16)
            all_photo_edges = photo_edges.reshape((-1, img_width, img_height, img_depth, 3))
            all_label_edges = label_edges.reshape((-1, img_width, img_height, img_depth, 3))
            if final_photo_edges is not None and final_label_edges is not None:
                final_photo_edges = np.concatenate([final_photo_edges, all_photo_edges], axis=0)
                final_label_edges = np.concatenate([final_label_edges, all_label_edges], axis=0)
            else:
                final_photo_edges = all_photo_edges
                final_label_edges = all_label_edges
    if extract_edges:
        return final_facade_photos, final_facade_labels, final_photo_edges, final_label_edges, results_dict
    else:
        return final_facade_photos, final_facade_labels, results_dict
    
    
def load_data_prediction(data_dir, img_width=256, img_height=256):

        # Get all .h5 files containing training images
    facade_photos_h5 = sorted(os.listdir(data_dir))
#     facade_labels_h5 = [f for f in os.listdir(os.path.join(data_dir_path, 'facades')) if '.h5' in f]

    final_facade_photos = None
    
    for index in range(len(facade_photos_h5)):
        facade_photos_path = data_dir+'/' + facade_photos_h5[index]
#         facade_labels_path = data_dir_path + '/facades/' + facade_labels_h5[index]
        facade_photos = nib.load(facade_photos_path).get_data()
        if facade_photos.any():
            facade_photos = cv2.resize(facade_photos, (img_width, img_height), interpolation=cv2.INTER_AREA)
            facade_photos, _ = normalize_array_max(facade_photos)
            # Resize and normalize images
    #         num_photos = facade_photos['data'].shape[0]
    #         num_labels = facade_labels['data'].shape[0]
        
    #         all_facades_photos = np.array(facade_photos['data'], dtype=np.float32)
            all_facades_photos = facade_photos.reshape((1, img_width, img_height, 1))
            
    #         all_facades_labels = np.array(facade_labels['data'], dtype=np.float32)
            
            if final_facade_photos is not None:
                        final_facade_photos = np.concatenate([final_facade_photos, all_facades_photos], axis=0)
            else:
                        final_facade_photos = all_facades_photos
    
    return final_facade_photos


def load_data_prediction_3D(data_dir, img_width=128, img_height=128, img_depth=128, mb=[2, 2, 2]):

        # Get all .h5 files containing training images
    facade_photos_h5 = sorted(os.listdir(os.path.join(data_dir)))

#     facade_labels_h5 = [f for f in os.listdir(os.path.join(data_dir_path, 'facades')) if '.h5' in f]

#    dx = 320
#    dy = 320
#    dz = 168

    dx = 240
    dy = 240
    dz = 155

    final_facade_photos = None
    
    diffX = dx - img_width
    diffY = dy - img_height
    diffZ = dz - img_depth

    while True:
        if diffX % (mb[0]-1) != 0:
            diffX += 1
            dx += 1
        elif diffY % (mb[1]-1) != 0:
            diffY += 1
            dy += 1
        elif diffZ % (mb[2]-1) != 0:
            diffZ += 1
            dz += 1
        else:
            break

    overlapX = diffX//(mb[0]-1)
    overlapY = diffY//(mb[1]-1)
    overlapZ = diffZ//(mb[2]-1)
    indX = [[x,x+img_width] for x in np.arange(0,dx,overlapX) if x+img_width<=dx]
    indY = [[x,x+img_height] for x in np.arange(0,dy,overlapY) if x+img_height<=dy]
    indZ = [[x,x+img_depth] for x in np.arange(0,dz,overlapZ) if x+img_depth<=dz]

    results_dict = {}
    
    for index in range(len(facade_photos_h5)):
        results_dict[index] = {}
        facade_photos_path = data_dir+'/' + facade_photos_h5[index]
        facade_photos_name = facade_photos_path.split('.')[0]+'_syn.nii.gz'
        results_dict[index]['name'] = facade_photos_name
#         facade_labels_path = data_dir_path + '/facades/' + facade_labels_h5[index]
        facade_photos_orig = nib.load(facade_photos_path).get_data()
        results_dict[index]['orig_dim'] = facade_photos_orig.shape
        results_dict[index]['orig_affine'] = nib.load(facade_photos_path).affine
        facade_photos_orig = resize(facade_photos_orig, (dx, dy, dz), order=3, mode='edge', cval=0,
                                    anti_aliasing=False)

        facades_photo = [facade_photos_orig[i[0]:i[1], j[0]:j[1], z[0]:z[1]] for z in indZ for j in indY for i in indX]
        facade_photos, max_photos = normalize_array_max(np.asarray(facades_photo))

        all_facades_photos = facade_photos.reshape((-1, img_width, img_height, img_depth, 1))
        results_dict[index]['max_photos'] = max_photos
        results_dict[index]['indexes'] = [indX, indY, indZ]
        results_dict[index]['im_size'] = [dx, dy, dz]

        if final_facade_photos is not None:
                    final_facade_photos = np.concatenate([final_facade_photos, all_facades_photos], axis=0)
        else:
                    final_facade_photos = all_facades_photos
    
    return final_facade_photos, results_dict

    
def gen_batch(X1, X2, batch_size, X3=None, X4=None):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        if X3 is not None and X4 is not None:
            yield X1[idx], X2[idx], X3[idx], X4[idx]
        else:
            yield X1[idx], X2[idx]


def get_disc_batch(X_full_batch, X_sketch_batch, generator_model, batch_counter, patch_size,
                   image_data_format, mode='3D', X_full_edge=None, label_smoothing=False, label_flipping=0, d3=False):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_full_batch)
        if X_full_edge is not None and mode=='3D':
            X_gen_edge = sobel_3D(X_disc)
            X_disc = np.concatenate([X_full_batch, X_disc, X_gen_edge], axis=-1)
        elif X_full_edge is not None and mode == '2D':
            X_gen_edge = sobel_2D(X_disc)
            X_disc = np.concatenate([X_full_batch, X_disc, X_gen_edge], axis=-1)
        else:
            X_disc = np.concatenate([X_full_batch, X_disc], axis=-1)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.float16)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0, high=0.3, size=y_disc.shape[0])
            y_disc[:, 0] = np.random.uniform(low=0.7, high=1.2, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        if X_full_edge is not None:
            X_disc = np.concatenate([X_full_batch, X_sketch_batch, X_full_edge], axis=-1)
        else:
            X_disc = np.concatenate([X_full_batch, X_sketch_batch], axis=-1)
#             X_disc = X_full_batch
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.float16)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.7, high=1.2, size=y_disc.shape[0])
            y_disc[:, 0] = np.random.uniform(low=0, high=0.3, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]
    
#     X_disc_edge = sobel_3D(X_disc[:, :, :, :, 0])
#     X_disc_edge = np.expand_dims(X_disc_edge, axis=-1)
#     X_disc = np.concatenate([X_disc, X_disc_edge], axis=-1)
    # Now extract patches form X_disc
    if mode == '3D':
        X_disc = extract_patches_3D(X_disc, image_data_format, patch_size)
    else:
        X_disc = extract_patches(X_disc, image_data_format, patch_size)

    return X_disc, y_disc


def plot_generated_batch(X_full, X_sketch, generator_model, num_epoch, dict_val=None, dset=None, d3=False):

    # Generate images
    X_gen = generator_model.predict(X_full)

    if d3:
        save_images_3D(X_full, X_sketch, X_gen, dict_val, num_epoch, dset)
    else:
        save_images(X_full, X_sketch, X_gen, num_epoch)



def save_images(real_images, real_sketches, generated_images, num_epoch):

    names = ['T1KM', 'FLAIR', 'FLAIR_gen']

    for i, image in enumerate([real_images, real_sketches, generated_images]):
        im2save = nib.Nifti1Image(image[0, :, :, 0], affine=np.eye(4))
        nib.save(im2save, '/mnt/sdb/logs_gan/results_im/{0}_epoch_{1}.nii.gz'.format(names[i], num_epoch))


def save_images_3D(real_images, real_sketches, generated_images, dict_val, num_epoch, dset):

    names = ['T1KM', 'FLAIR', 'FLAIR_gen']
    im_shape = dict_val['im_size']
    indexes = dict_val['indexes']
    for n_image, image in enumerate([real_images, real_sketches, generated_images]):
        final_image = np.zeros((im_shape[0], im_shape[1], im_shape[2], generated_images.shape[0]))-2
        k = 0
        for z in indexes[2]:
            for j in indexes[1]:
                for i in indexes[0]:
                    final_image[i[0]:i[1],j[0]:j[1],z[0]:z[1],k] = image[k,:,:,:,0]
                    k += 1
        final_image[final_image==-2] = np.nan
        final_image = np.nanmean(final_image, axis=3)
        im2save = nib.Nifti1Image(final_image, affine=np.eye(4))
        nib.save(im2save, '{2}/results_im/{0}_epoch_{1}.nii.gz'.format(names[n_image], num_epoch, dset))


def save_prediction_3D(generated_images, dict_val):
    
    n_images = len(dict_val)
    batches = generated_images.shape[0]//n_images
    for n in range(n_images):
        im_shape = dict_val[n]['im_size']
        indexes = dict_val[n]['indexes']
        image = generated_images[n*batches:(n+1)*batches, :]
        final_image = np.zeros((im_shape[0], im_shape[1], im_shape[2], generated_images.shape[0]//n_images), dtype=np.float16)-2
        k = 0
        for z in indexes[2]:
            for j in indexes[1]:
                for i in indexes[0]:
                    final_image[i[0]:i[1],j[0]:j[1],z[0]:z[1],k] = image[k,:,:,:,0]
                    k += 1
        final_image[final_image==-2] = np.nan
        final_image = np.nanmean(final_image, axis=3)
        original_dim = dict_val[n]['orig_dim']
        final_image = resize(final_image.astype(np.float32), (original_dim[0], original_dim[1], original_dim[2]), order=3, mode='edge', cval=0,
                                    anti_aliasing=False)
        final_image = inverse_normalize_array_max(final_image.astype(np.float32), dict_val[n]['max_photos'])
        im2save = nib.Nifti1Image(final_image, affine=dict_val[n]['orig_affine'])
        nib.save(im2save, dict_val[n]['name'])
