from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from .utils.filemanip import get_nifti
from .utils.mouse_segmentation import preprocessing
import nibabel as nib
import os
import glob
from dl.utils.general_utils import normalize_array_max, sobel_3D, sobel_2D, normalize_min_max
from skimage.transform import resize
from array import array


def data_generator(x_train, y_train, batch_size, seed=42):

    data_generator = ImageDataGenerator().flow(x_train, x_train, batch_size, seed=seed)
    mask_generator = ImageDataGenerator().flow(y_train, y_train, batch_size, seed=seed)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()

        yield x_batch, y_batch


def image_generator(files, batch_size = 64):
    
    while True:
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a = files, size = batch_size)
        batch_input = []
        batch_output = [] 
        
        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
#             with open('/home/fsforazz/Desktop/used_images.txt', 'a') as f:
#                 f.write(input_path+'\n')
            image = get_nifti(input_path)
            mask = get_nifti(input_path, labels=True)
          
#             image = preprocess_input(image.astype('float64'), mode='tf')
            image = preprocessing(image)
            mask = preprocessing(mask, label=True)
            batch_input += [image]
            batch_output += [mask]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        
        yield(batch_x, batch_y)


def data_prep_train_on_batch(files, method='mouse_fibrosis'):
    
    images = []
    labels = []

    for file in files:

        image = get_nifti(file)
        mask = get_nifti(file, labels=True, method=method)

        image = preprocessing(image, method=method)
        mask = preprocessing(mask, label=True, method=method)
        images += [image]
        labels += [mask]
    
    return np.array(images), np.array(labels)


def load_data_3D(data_dir, data_type, mb=[], bs=None, init=None, extract_edges=True, prediction=False,
                 img_size=(240, 240, 155), patch_size=(128, 128, 128), binarize=False, normalize=True):

    if bs is not None and init is not None:
        facade_photos_h5 = sorted(glob.glob(os.path.join(data_dir, data_type)))[init:bs]
    else:
        facade_photos_h5 = sorted(glob.glob(os.path.join(data_dir, data_type)))

    dx = img_size[0]
    dy = img_size[1]
    dz = img_size[2]
    
    img_width = patch_size[0]
    img_height = patch_size[1]
    img_depth = patch_size[2]

    if len(mb) < 3:
        mb.append((img_size[0]//img_width)+1)
    if dx < img_width:
        img_width = dx
    if len(mb) < 3:
        mb.append((img_size[1]//img_height)+1)
    if dy < img_height:
        img_height = dy
    if len(mb) < 3:
        mb.append((img_size[2]//img_depth)+1)
    if dz <= img_depth:
        img_depth = dz
        mb[-1] = 1  

    final_facade_photos = None
    final_photo_edges = None
    
    diffX = dx - img_width
    diffY = dy - img_height
    diffZ = dz - img_depth

#     while True:
#         if diffX % (mb[0]-1) != 0:
#             diffX += 1
#             dx += 1
#         elif diffY % (mb[1]-1) != 0:
#             diffY += 1
#             dy += 1
#         elif diffZ % (mb[2]-1) != 0:
#             diffZ += 1
#             dz += 1
#         else:
#             break
    try:
        overlapX = diffX//(mb[0]-1)
    except ZeroDivisionError:
        overlapX = dx
    try:
        overlapY = diffY//(mb[1]-1)
    except ZeroDivisionError:
        overlapY = dy
    try:
        overlapZ = diffZ//(mb[2]-1)
    except ZeroDivisionError:
        overlapZ = dz
    indX = [[x, x+img_width] for x in np.arange(0, dx, overlapX) if x+img_width<=dx]
    indY = [[x, x+img_height] for x in np.arange(0, dy, overlapY) if x+img_height<=dy]
    indZ = [[x, x+img_depth] for x in np.arange(0, dz, overlapZ) if x+img_depth<=dz]

    for index in range(len(facade_photos_h5)):

        facade_photos_path = facade_photos_h5[index]
        facade_photos_orig = nib.load(facade_photos_path).get_data()
        original_size = facade_photos_orig.shape
        if facade_photos_orig.shape != img_size:
            facade_photos_orig = resize(facade_photos_orig, (dx, dy, dz), order=3, mode='edge', cval=0,
                                        anti_aliasing=False)
        
        facade_photos = [facade_photos_orig[i[0]:i[1], j[0]:j[1], z[0]:z[1]] for z in indZ for j in indY for i in indX]
        facade_photos = np.asarray(facade_photos, dtype=np.float16)
        if normalize:
            facade_photos, _ = normalize_array_max(facade_photos)
        if binarize:
            facade_photos[facade_photos != 0] = 1
        

        all_facades_photos = facade_photos.reshape((-1, img_width, img_height, img_depth, 1))

        if final_facade_photos is not None:
                    final_facade_photos = np.concatenate([final_facade_photos, all_facades_photos], axis=0)
        else:
                    final_facade_photos = all_facades_photos
        
        results_dict = {}
        if prediction:
            results_dict[index] = {}
            facade_photos_name = facade_photos_path.split('.')[0]+'_syn.nii.gz'
            results_dict[index]['name'] = facade_photos_name
            results_dict[index]['orig_dim'] = original_size
            results_dict[index]['orig_affine'] = nib.load(facade_photos_path).affine
            results_dict[index]['indexes'] = [indX, indY, indZ]
            results_dict[index]['im_size'] = [dx, dy, dz]

        if extract_edges:
            facade_photos_edge = sobel_3D(facade_photos_orig)
            facades_photo_edge = [facade_photos_edge[i[0]:i[1], j[0]:j[1], z[0]:z[1]] for z in indZ for j in indY for i in indX]
            photo_edges = np.asarray(facades_photo_edge, dtype=np.float16)
            all_photo_edges = photo_edges.reshape((-1, img_width, img_height, img_depth, 3))
            if final_photo_edges is not None:
                final_photo_edges = np.concatenate([final_photo_edges, all_photo_edges], axis=0)
            else:
                final_photo_edges = all_photo_edges

    if final_photo_edges is not None:
        final_facade_photos = np.concatenate([final_facade_photos, final_photo_edges], axis=-1)

    if prediction:
        return final_facade_photos, results_dict
    else:
        return final_facade_photos


def load_data_2D(data_dir, data_type, data_list=[], array=None, mb=[], bs=None, init=None, extract_edges=True, prediction=False,
                 img_size=(192, 192), patch_size=(96, 96), binarize=False, normalize=True):
    if array is not None:
        facade_photos_h5 = [1]
    else:
        if data_list:
            facade_photos_h5 = data_list
        elif bs is not None and init is not None:
            facade_photos_h5 = sorted(glob.glob(os.path.join(data_dir, data_type)))[init:bs]
        else:
            facade_photos_h5 = sorted(glob.glob(os.path.join(data_dir, data_type)))

    dx = img_size[0]
    dy = img_size[1]
    
    img_width = patch_size[0]
    img_height = patch_size[1]

    if not mb:
        if img_size[0] % patch_size[0] == 0:
            mb.append((img_size[0]//patch_size[0]))
        else:
            mb.append((img_size[0]//patch_size[0])+1)
        if img_size[1] % patch_size[1] == 0:
            mb.append((img_size[1]//patch_size[1]))
        else:
            mb.append((img_size[1]//patch_size[1])+1)

    final_facade_photos = None
    final_photo_edges = None
    
    diffX = dx - img_width
    diffY = dy - img_height

    overlapX = diffX//(mb[0]-1)
    overlapY = diffY//(mb[1]-1)
    indX = [[x, x+img_width] for x in np.arange(0, dx, overlapX) if x+img_width<=dx]
    indY = [[x, x+img_height] for x in np.arange(0, dy, overlapY) if x+img_height<=dy]

    for index in range(len(facade_photos_h5)):

        if array is not None:
            facade_photos_orig = array
        else:
            facade_photos_path = facade_photos_h5[index]
            facade_photos_orig = nib.load(facade_photos_path).get_data()
        original_size = facade_photos_orig.shape
        if facade_photos_orig.shape != img_size:
            facade_photos_orig = resize(facade_photos_orig, (dx, dy), order=3, mode='edge', cval=0,
                                        anti_aliasing=False)
        
        facade_photos = [facade_photos_orig[i[0]:i[1], j[0]:j[1]] for j in indY for i in indX]
        facade_photos = np.asarray(facade_photos, dtype=np.float16)
        if normalize:
            facade_photos = normalize_min_max(facade_photos)
        if binarize:
            facade_photos[facade_photos != 0] = 1
        

        all_facades_photos = facade_photos.reshape((-1, img_width, img_height, 1))

        if final_facade_photos is not None:
                    final_facade_photos = np.concatenate([final_facade_photos, all_facades_photos], axis=0)
        else:
                    final_facade_photos = all_facades_photos
        
        results_dict = {}
        if prediction:
            results_dict[index] = {}
            results_dict[index]['orig_dim'] = original_size
            results_dict[index]['indexes'] = [indX, indY]
            results_dict[index]['im_size'] = [dx, dy]

        if extract_edges:
            facade_photos_edge = sobel_3D(facade_photos_orig)
            facades_photo_edge = [facade_photos_edge[i[0]:i[1], j[0]:j[1]] for j in indY for i in indX]
            photo_edges = np.asarray(facades_photo_edge, dtype=np.float16)
            all_photo_edges = photo_edges.reshape((-1, img_width, img_height, 3))
            if final_photo_edges is not None:
                final_photo_edges = np.concatenate([final_photo_edges, all_photo_edges], axis=0)
            else:
                final_photo_edges = all_photo_edges

    if final_photo_edges is not None:
        final_facade_photos = np.concatenate([final_facade_photos, final_photo_edges], axis=-1)

    if prediction:
        return final_facade_photos, results_dict, final_facade_photos.shape[0]
    else:
        return final_facade_photos


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


def get_disc_batch(X_full_batch, X_sketch_batch, generator_model, batch_counter, patch_size,
                   image_data_format, mode='3D', label_smoothing=False, label_flipping=0, use_edge=False):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_full_batch = X_full_batch[:, :, :, :, 0].reshape([-1, 128, 128, 128, 1])
        X_disc = generator_model.predict(X_full_batch)
        if use_edge is not None and mode=='3D':
            X_gen_edge = sobel_3D(X_disc)
            X_disc = np.concatenate([X_full_batch,  X_gen_edge, X_disc], axis=-1)
        elif use_edge is not None and mode == '2D':
            X_gen_edge = sobel_2D(X_disc)
            X_disc = np.concatenate([X_full_batch,  X_gen_edge, X_disc], axis=-1)
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

        X_disc = np.concatenate([X_full_batch, X_sketch_batch], axis=-1)
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

    # Now extract patches form X_disc
    if mode == '3D':
        X_disc = extract_patches_3D(X_disc, image_data_format, patch_size)
    else:
        X_disc = extract_patches(X_disc, image_data_format, patch_size)

    return X_disc, y_disc
