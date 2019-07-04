from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import optimizers as opt
from functools import partial


def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor)

def loss_dice_coefficient_error(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

# K.set_image_data_format("channels_last")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    K.set_image_data_format("channels_first")
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

#     if include_label_wise_dice_coefficients and n_labels > 1:
#         label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
#         if metrics:
#             metrics = metrics + label_wise_dice_metrics
#         else:
#             metrics = label_wise_dice_metrics

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss_dice_coefficient_error, metrics=metrics)
    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

def mouse_lung_seg(pretrained_weights = None, input_size = (96,96,1)):

    inputs = Input(input_size)
    conv1 = Conv2D(48, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(48, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(96, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(96, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(192, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(192, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(384, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(384, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
#     drop4 = Dropout(0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(768, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(768, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
#     drop5 = Dropout(0.1)(conv5)

    up6 = Conv2D(384, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    up6 = BatchNormalization()(up6)
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(384, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(384, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(192, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(192, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(192, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(96, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(96, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(96, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(48, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(48, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(48, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

#     model.compile(optimizer = Adam(lr = 1e-4), loss ='binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n


def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m


def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False,
         pretrained_weights=None, mode='regression', init_lr=0.0001):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)

    if mode == 'classification':
        if out_ch == 1:
            outputs = Conv2D(filters=out_ch, kernel_size=(1,1), 
                            activation='sigmoid')(o)
        else:
            outputs = Conv2D(filters=out_ch, kernel_size=(1,1), 
                            activation='softmax')(o)

        unet_model = Model(inputs=i, outputs=outputs)

        if out_ch == 1:
            unet_model.compile(loss=loss_dice_coefficient_error, 
                                optimizer=Adam(lr=init_lr), metrics=[dice_coefficient])
        else:
            unet_model.compile(loss='categorical_crossentropy', 
                                optimizer=Adam(lr=init_lr), metrics=['accuracy', 'categorical_crossentropy'])
    elif mode =='regression':
        outputs = Conv2D(filters=out_ch, kernel_size=(1,1), 
                        activation='tanh')(o)
        unet_model = Model(inputs=i, outputs=outputs)
        unet_model.compile(loss='mse', optimizer=Adam(lr=init_lr))
    else:
        raise ValueError('mode must be either `classification` or `regression`')
#     o = Conv2D(out_ch, 1, activation='sigmoid')(o)
#     model = Model(inputs=i, outputs=o)
    
    if(pretrained_weights):
        unet_model.load_weights(pretrained_weights)

    return unet_model


def create_unet_model3D(input_image_size,
                        n_labels=1,
                        layers=4,
                        lowest_resolution=16,
                        convolution_kernel_size=(5,5,5),
                        deconvolution_kernel_size=(5,5,5),
                        pool_size=(2,2,2),
                        strides=(2,2,2),
                        mode='classification',
                        output_activation='tanh',
                        init_lr=0.0001,
                        pretrained_weights=None):
    """
    Create a 3D Unet model
    Example
    -------
    unet_model = create_unet_model3D( (128,128,128,1), 1, 4)
    """
    K.set_image_data_format("channels_last")
    layers = np.arange(layers)
    number_of_classification_labels = n_labels
    
    inputs = Input(shape=input_image_size)

    ## ENCODING PATH ##

    encoding_convolution_layers = []
    pool = None
    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2**(layers[i])

        if i == 0:
            conv = Conv3D(filters=number_of_filters, 
                            kernel_size=convolution_kernel_size,
                            activation='relu',
                            padding='same')(inputs)
        else:
            conv = Conv3D(filters=number_of_filters, 
                            kernel_size=convolution_kernel_size,
                            activation='relu',
                            padding='same')(pool)

        encoding_convolution_layers.append(Conv3D(filters=number_of_filters, 
                                                        kernel_size=convolution_kernel_size,
                                                        activation='relu',
                                                        padding='same')(conv))

        if i < len(layers)-1:
            pool = MaxPooling3D(pool_size=pool_size)(encoding_convolution_layers[i])

    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers)-1]
    for i in range(1,len(layers)):
        number_of_filters = lowest_resolution * 2**(len(layers)-layers[i]-1)
        tmp_deconv = Conv3DTranspose(filters=number_of_filters, kernel_size=deconvolution_kernel_size,
                                     padding='same')(outputs)
        tmp_deconv = UpSampling3D(size=pool_size)(tmp_deconv)
        outputs = Concatenate(axis=4)([tmp_deconv, encoding_convolution_layers[len(layers)-i-1]])

        outputs = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation='relu', padding='same')(outputs)
        outputs = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation='relu', padding='same')(outputs)

    if mode == 'classification':
        if number_of_classification_labels == 1:
            outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1,1,1), 
                            activation='sigmoid')(outputs)
        else:
            outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1,1,1), 
                            activation='softmax')(outputs)

        unet_model = Model(inputs=inputs, outputs=outputs)

        if number_of_classification_labels == 1:
            unet_model.compile(loss=loss_dice_coefficient_error, 
                                optimizer=opt.Adam(lr=init_lr), metrics=[dice_coefficient])
        else:
            unet_model.compile(loss='categorical_crossentropy', 
                                optimizer=opt.Adam(lr=init_lr), metrics=['accuracy', 'categorical_crossentropy'])
    elif mode =='regression':
        outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1,1,1), 
                        activation=output_activation)(outputs)
        unet_model = Model(inputs=inputs, outputs=outputs)
#         unet_model.compile(loss='mse', optimizer=opt.Adam(lr=init_lr))
    else:
        raise ValueError('mode must be either `classification` or `regression`')
    
    if(pretrained_weights):
        unet_model.load_weights(pretrained_weights)

    return unet_model


def create_unet_model2D(input_image_size,
                        n_labels=1,
                        layers=4,
                        lowest_resolution=16,
                        convolution_kernel_size=(5,5),
                        deconvolution_kernel_size=(5,5),
                        pool_size=(2,2),
                        strides=(2,2),
                        mode='classification',
                        output_activation='tanh',
                        init_lr=0.0001,
                        pretrained_weights=None):
    """
    Create a 2D Unet model

    Example
    -------
    unet_model = create_Unet_model2D( (100,100,1), 1, 4)
    """
    layers = np.arange(layers)
    number_of_classification_labels = n_labels
    
    inputs = Input(shape=input_image_size)

    ## ENCODING PATH ##

    encoding_convolution_layers = []
    pool = None
    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2**(layers[i])

        if i == 0:
            conv = Conv2D(filters=number_of_filters, 
                                kernel_size=convolution_kernel_size,
                                activation='relu',
                                padding='same')(inputs)
        else:
            conv = Conv2D(filters=number_of_filters, 
                                kernel_size=convolution_kernel_size,
                                activation='relu',
                                padding='same')(pool)

        encoding_convolution_layers.append(Conv2D(filters=number_of_filters, 
                                                        kernel_size=convolution_kernel_size,
                                                        activation='relu',
                                                        padding='same')(conv))

        if i < len(layers)-1:
            pool = MaxPooling2D(pool_size=pool_size)(encoding_convolution_layers[i])

    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers)-1]
    for i in range(1,len(layers)):
        number_of_filters = lowest_resolution * 2**(len(layers)-layers[i]-1)
        tmp_deconv = Conv2DTranspose(filters=number_of_filters, kernel_size=deconvolution_kernel_size,
                                     padding='same')(outputs)
        tmp_deconv = UpSampling2D(size=pool_size)(tmp_deconv)
        outputs = Concatenate(axis=3)([tmp_deconv, encoding_convolution_layers[len(layers)-i-1]])

        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation='relu', padding='same')(outputs)
        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation='relu', padding='same')(outputs)

    if mode == 'classification':
        if number_of_classification_labels == 1:
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1), 
                            activation='sigmoid')(outputs)
        else:
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1), 
                            activation='softmax')(outputs)

        unet_model = Model(inputs=inputs, outputs=outputs)

#         if number_of_classification_labels == 1:
#             unet_model.compile(loss=loss_dice_coefficient_error, 
#                                 optimizer=opt.Adam(lr=init_lr), metrics=[dice_coefficient])
#         else:
#             unet_model.compile(loss='categorical_crossentropy', 
#                                 optimizer=opt.Adam(lr=init_lr), metrics=['accuracy', 'categorical_crossentropy'])
    elif mode =='regression':
        outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1), 
                        activation=output_activation)(outputs)
        unet_model = Model(inputs=inputs, outputs=outputs)
#         unet_model.compile(loss='mse', optimizer=opt.Adam(lr=init_lr))
    else:
        raise ValueError('mode must be either `classification` or `regression`')
    
    if(pretrained_weights):
        unet_model.load_weights(pretrained_weights)

    return unet_model

