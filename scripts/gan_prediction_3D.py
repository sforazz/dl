import scripts.models as models
import dl.utils.data_utils as data_utils
import numpy as np


def prediction(**kwargs):
    """
    Train model
    Load the whole train data in memory for faster operations
    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    generator = kwargs["generator"]
    image_data_format = kwargs["image_data_format"]
    patch_size = kwargs["patch_size"]
    bn_mode = kwargs["bn_mode"]
    dset = kwargs["dset"]
    use_mbd = kwargs["use_mbd"]
    weights = kwargs["weights"]

    X_full_val, val_dict = data_utils.load_data_prediction_3D(dset)
    img_dim = X_full_val.shape[-4:]

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, _ = data_utils.get_nb_patch(img_dim, patch_size, image_data_format)

    try:

        # Load generator model
        generator_model = models.load("generator_unet_3D_%s" % generator,
                                      img_dim,
                                      nb_patch,
                                      bn_mode,
                                      use_mbd,
                                      batch_size,
                                      False)
        generator_model.load_weights(weights)
#         generator_model = build_unet_generator()
        # Load discriminator model
        X_gen = []
        for i in range(X_full_val.shape[0]):
            gen = generator_model.predict(X_full_val[i, :].reshape(1, X_full_val.shape[1], X_full_val.shape[2], X_full_val.shape[3], 1))
            X_gen.append(gen[0, :])
        X_gen = np.asarray(X_gen)
        data_utils.save_prediction_3D(X_gen, val_dict)
        print('Predicted')

    except KeyboardInterrupt:
        pass


def launch_prediction(**kwargs):

    # Launch prediction
    prediction(**kwargs)


d_params = {"dset": '/mnt/sdb/data_T1_to_FLAIR_normalized_new/test_3D', #"/mnt/sdb/brats_normalized/",
            "generator": 'upsampling',
            "batch_size": 3,
            "image_data_format": "channels_last",
            "bn_mode": 2,
            "img_dim": 256,
            "patch_size": (32, 32),
            "use_mbd": True,
            #"weights": '/mnt/sdb/data_T1_to_FLAIR_normalized_new/gen_weights_epoch40_no_sobel.h5'
             "weights": '/mnt/sdb/data_T1KM_to_FLAIR/gen_weights_epoch200_multi_ds.h5'
            }

launch_prediction(**d_params)