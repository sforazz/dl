import os
import time
import numpy as np
import scripts.models as models
from keras.utils import generic_utils
from keras.optimizers import Adam
import keras.backend as K
import dl.utils.general_utils as general_utils
import dl.utils.data_utils as data_utils
from keras.callbacks import TensorBoard
import random


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


def train(**kwargs):
    """
    Train model
    Load the whole train data in memory for faster operations
    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    model_name = kwargs["model_name"]
    generator = kwargs["generator"]
    image_data_format = kwargs["image_data_format"]
    patch_size = kwargs["patch_size"]
    bn_mode = kwargs["bn_mode"]
    label_smoothing = kwargs["use_label_smoothing"]
    label_flipping = kwargs["label_flipping"]
    dset = kwargs["dset"]
    use_mbd = kwargs["use_mbd"]
    do_plot = kwargs["do_plot"]
    logging_dir = kwargs["logging_dir"]
    save_every_epoch = kwargs["epoch"]
    use_generator = kwargs["use_generator"]
    img_dim = kwargs["img_dim"]
    
    epoch_size = n_batch_per_epoch * batch_size * 5
    lr_init = 1E-5

    # Setup environment (logging directory etc)
    general_utils.setup_logging(model_name, logging_dir=logging_dir)
    
    tensorboard = TensorBoard(log_dir=os.path.join(logging_dir, "logs/tensorboard").format(time.time()))

    # Load and rescale data
    if not use_generator:
        try:
            X_full_train = np.load(os.path.join(dset, 'training_T1.npy'))
            X_sketch_train = np.load(os.path.join(dset, 'training_FLAIR.npy'))
        except:
            X_full_train, X_sketch_train, _ = data_utils.load_data_3D(dset, 'train', image_data_format,
                                                                      extract_edges=False)
            np.save(os.path.join(dset, 'training_T1.npy'), X_full_train)
            np.save(os.path.join(dset, 'training_FLAIR.npy'), X_sketch_train)
        img_dim = X_full_train.shape[-4:]
    
        # Get the number of non overlapping patch and the size of input image to the discriminator
        nb_patch, img_dim_disc = data_utils.get_nb_patch_3D(img_dim, patch_size, image_data_format)
    else:
        try:
            data_full = [x for x in os.listdir(dset) if x.endswith('.npy') and 'T1' in x and 'edge' not in x]
            data_sketch = [x for x in os.listdir(dset) if x.endswith('.npy') and 'FLAIR' in x and 'edge' not in x]
            img_dim = img_dim
            print('Found {} parts of data'.format(len(data_full)))
            epoch_size = n_batch_per_epoch * batch_size * len(data_full)
            img_dim_disc = (img_dim[0], img_dim[1], img_dim[2], 2)
            nb_patch, img_dim_disc = data_utils.get_nb_patch_3D(img_dim_disc, patch_size, image_data_format)
        except:
            raise Exception('If you use data generator you must specify the image dimensions (for example, (128, 128, 128, 1)).')

    try:

        # Create optimizers
        opt_dcgan = Adam(lr=lr_init, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        opt_discriminator = Adam(lr=lr_init, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # Load generator model
        generator_model = models.load("generator_unet_3D_%s" % generator,
                                      img_dim,
                                      nb_patch,
                                      bn_mode,
                                      use_mbd,
                                      batch_size,
                                      do_plot)
#         generator_model = build_unet_generator()

        # Load discriminator model
        discriminator_model = models.load("DCGAN_discriminator_3D",
                                          img_dim_disc,
                                          nb_patch,
                                          bn_mode,
                                          use_mbd,
                                          batch_size,
                                          do_plot)
        
        generator_model.compile(loss='mae', optimizer=opt_discriminator)
#         generator_model.load_weights('/data/logs_gan/models/3D_lf=0_ps=32_bs=2_no_sobel_2channels_ds/gen_weights_epoch200.h5')
        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN_3D_no_sobel_2(generator_model,
                                               discriminator_model,
                                               img_dim,
                                               patch_size,
                                               image_data_format)

#         loss = [l1_loss, 'binary_crossentropy']
#         loss_weights = [3E1, 1]
        loss = [l1_loss, 'binary_crossentropy']
        loss_weights = [3E2, 1]
        DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)
#         DCGAN_model.load_weights('/data/logs_gan/models/3D_lf=0_ps=32_bs=2_no_sobel_2channels_ds/DCGAN_weights_epoch200.h5')
        discriminator_model.trainable = True
        discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)
#         discriminator_model.load_weights('/data/logs_gan/models/3D_lf=0_ps=32_bs=2_no_sobel_2channels_ds/disc_weights_epoch200.h5')

        gen_loss = 100
        disc_loss = 100

        tensorboard.set_model(generator_model)
        tensorboard.set_model(discriminator_model)
        # Start training
        print("Start training")
#         generator_model.load_weights('/mnt/sdb/logs_gan/models/CNN/gen_weights_epoch390.h5')
#         res=generator_model.predict(X_sketch_val)

        decay = 1
#         z = 0
#         init = 0
        for e in range(nb_epoch):
            if use_generator:
                data = list(zip(data_full, data_sketch))
                random.shuffle(data)
            else:
                data = [[X_full_train, X_sketch_train]]

            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            
            start = time.time()
            dis_losses = []
            gen_losses = []
            if e > 200:
                lr = lr_init - 0.000001*decay
                if lr < 0:
                    lr = 2.710505431213761e-20
                else:
                    decay += 1
                K.set_value(generator_model.optimizer.lr, lr)
                K.set_value(DCGAN_model.optimizer.lr, lr)
                K.set_value(discriminator_model.optimizer.lr, lr)
            if e == 0 or e > 200:
                print('Generator LR: {}'.format(K.get_value(generator_model.optimizer.lr)))
                print('DCGAN LR: {}'.format(K.get_value(DCGAN_model.optimizer.lr)))
                print('Discriminator LR: {}'.format(K.get_value(discriminator_model.optimizer.lr)))

            for f, s in data:
                if use_generator:
                    X_full_train = np.load(os.path.join(dset, f))
                    X_sketch_train = np.load(os.path.join(dset, s))
                else:
                    X_full_train = f
                    X_sketch_train = s
                batch_counter = 1
                for X_full_batch, X_sketch_batch in data_utils.gen_batch(X_full_train, X_sketch_train, batch_size):
    
                    # Create a batch to feed the discriminator model
                    X_disc, y_disc = data_utils.get_disc_batch(X_full_batch,
                                                               X_sketch_batch,
                                                               generator_model,
                                                               batch_counter,
                                                               patch_size,
                                                               image_data_format,
                                                               label_smoothing=label_smoothing,
                                                               label_flipping=label_flipping,
                                                               mode='3D')
    
                    # Update the discriminator
                    disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)
                    # Create a batch to feed the generator model
                    X_gen, X_gen_target = next(data_utils.gen_batch(X_full_train, X_sketch_train, batch_size))
                    y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
                    y_gen[:, 1] = 1
    
                    # Freeze the discriminator
                    discriminator_model.trainable = False
                    gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
                    # Unfreeze the discriminator
                    discriminator_model.trainable = True
                    
                    gen_losses.append(gen_loss[0])
                    dis_losses.append(disc_loss)
                    batch_counter += 1
                    progbar.add(batch_size, values=[("D logloss", disc_loss),
                                                    ("G tot", gen_loss[0]),
                                                    ("G L1", gen_loss[1]),
                                                    ("G logloss", gen_loss[2])])
    
                    if batch_counter >= n_batch_per_epoch:
                        break
            general_utils.write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), e)
            general_utils.write_log(tensorboard, 'generator_loss', np.mean(gen_losses), e)
            print("")
            print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

            if e % save_every_epoch == 0:
                gen_weights_path = os.path.join(logging_dir, 'models/%s/gen_weights_epoch%s.h5' % (model_name, e))
                generator_model.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join(logging_dir, 'models/%s/disc_weights_epoch%s.h5' % (model_name, e))
                discriminator_model.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join(logging_dir, 'models/%s/DCGAN_weights_epoch%s.h5' % (model_name, e))
                DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)

    except KeyboardInterrupt:
        pass

def launch_training(**kwargs):

    # Launch training
    train(**kwargs)


d_params = {"dset": "/data/gan_data_bc_T1/",#"/mnt/sdb/data_T1_to_FLAIR_normalized/", #
            "generator": 'upsampling',
            "batch_size": 2,
            "n_batch_per_epoch": 100,
            "nb_epoch": 401,
            "model_name": "3D_lf=0_ps=32_bs=2_no_sobel_2channels_e=400",
            "epoch": 10,
            "nb_classes": 1,
            "do_plot": False,
            "image_data_format": "channels_last",
            "bn_mode": 2,
            "img_dim": (128, 128, 128, 1),
            "use_label_smoothing": True,
            "label_flipping": 0,
            "patch_size": (32, 32, 32),
            "use_mbd": True,
            "logging_dir": "/data/logs_gan_T1/", #'/mnt/sdb/logs_gan/',  
            "use_generator": True
            }

launch_training(**d_params)
