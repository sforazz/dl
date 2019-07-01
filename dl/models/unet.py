from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import optimizers as opt


def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor)

def loss_dice_coefficient_error(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def generator(pretrained_weights = None, input_size = (128, 128, 128, 1)):

    inputs = Input(input_size)
    conv1 = Conv3D(64, 4, strides=2, padding = 'same')(inputs)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = Conv3D(64, 4, strides=1, padding = 'same')(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv2 = Conv3D(128, 4, strides=2, padding = 'same')(conv1)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(128, 4, strides=1, padding = 'same')(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv3D(256, 4, strides=2, padding = 'same')(conv2)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(256, 4, strides=1, padding = 'same')(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv3D(512, 4, strides=2, padding = 'same')(conv3)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv3D(512, 4, strides=1, padding = 'same')(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv5 = Conv3D(512, 4, strides=2, padding = 'same')(conv4)
    conv5 = LeakyReLU(alpha=0.2)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3D(512, 4, strides=1, padding = 'same')(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv3D(512, 4, strides=2, padding = 'same')(conv5)
    conv6 = LeakyReLU(alpha=0.2)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3D(512, 4, strides=1, padding = 'same')(conv6)
    conv6 = LeakyReLU(alpha=0.2)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv7 = Conv3D(512, 4, strides=2, padding = 'same')(conv6)
    conv7 = LeakyReLU(alpha=0.2)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3D(512, 4, strides=1, padding = 'same')(conv7)
    conv7 = LeakyReLU(alpha=0.2)(conv7)
    conv7 = BatchNormalization()(conv7)

    up1 = Conv3DTranspose(filters=512, kernel_size=4, padding='same')(conv7)
    up1 = UpSampling3D(size=2)(up1)
    up1 = Concatenate(axis=4)([up1, conv6])
    up1 = Conv3D(512, 4, padding = 'same', activation='relu')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Conv3D(512, 4, padding = 'same', activation='relu')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Dropout(0.5)(up1)
    up2 = Conv3DTranspose(filters=512, kernel_size=4, padding='same')(up1)
    up2 = UpSampling3D(size=2)(up2)
    up2 = Concatenate(axis=4)([up2, conv5])
    up2 = Conv3D(512, 4, padding = 'same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Conv3D(512, 4, padding = 'same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Dropout(0.5)(up2)
    up3 = Conv3DTranspose(filters=512, kernel_size=4, padding='same')(up2)
    up3 = UpSampling3D(size=2)(up3)
    up3 = Concatenate(axis=4)([up3, conv4])
    up3 = Conv3D(512, 4, padding = 'same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Conv3D(512, 4, padding = 'same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Dropout(0.5)(up3)
    up4 = Conv3DTranspose(filters=256, kernel_size=4, padding='same')(up3)
    up4 = UpSampling3D(size=2)(up4)
    up4 = Concatenate(axis=4)([up4, conv3])
    up4 = Conv3D(256, 4, padding = 'same', activation='relu')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Conv3D(256, 4, padding = 'same', activation='relu')(up4)
    up4 = BatchNormalization()(up4)
    up5 = Conv3DTranspose(filters=128, kernel_size=4, padding='same')(up4)
    up5 = UpSampling3D(size=2)(up5)
    up5 = Concatenate(axis=4)([up5, conv2])
    up5 = Conv3D(128, 4, padding = 'same', activation='relu')(up5)
    up5 = BatchNormalization()(up5)
    up5 = Conv3D(128, 4, padding = 'same', activation='relu')(up5)
    up5 = BatchNormalization()(up5)
    up6 = Conv3DTranspose(filters=64, kernel_size=4, padding='same')(up5)
    up6 = UpSampling3D(size=2)(up6)
    up6 = Concatenate(axis=4)([up6, conv1])
    up6 = Conv3D(64, 4,  padding = 'same', activation='relu')(up6)
    up6 = BatchNormalization()(up6)
    up6 = Conv3D(64, 4, padding = 'same', activation='relu')(up6)
    up6 = BatchNormalization()(up6)
    up7 = UpSampling3D(size=2)(up6)

    outputs = Conv3D(filters=1, kernel_size=1, activation='tanh')(up7)

    model = Model(input = inputs, output = outputs)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

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

