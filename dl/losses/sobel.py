import keras.backend as K


sobelFilter_3D = K.variable([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], 
                           [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                          [[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], 
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[1, 2, 1], [2, 4, 2], [1, 2, 1]]],
                          [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                           [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]])
sobelFilter_3D = K.expand_dims(sobelFilter_3D, axis=-2)
sobelFilter_3D = K.cast(sobelFilter_3D, K.tf.float16)

#this contains both X and Y sobel filters in the format (3,3,1,2)
#size is 3 x 3, it considers 1 input channel and has two output channels: X and Y
sobelFilter = K.variable([[[[1.,  1.]], [[0.,  2.]],[[-1.,  1.]]],
                          [[[2.,  0.]], [[0.,  0.]],[[-2.,  0.]]],
                          [[[1., -1.]], [[0., -2.]],[[-1., -1.]]]])

def expandedSobel(inputTensor):

    #this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return sobelFilter * inputChannels


def expandedSobel3D(inputTensor):

    #this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0, 0, 0, 0, :]), (1, 1, 1, -1, 1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return sobelFilter_3D * inputChannels


def sobelLoss(yTrue,yPred):

    #get the sobel filter repeated for each input channel
    filt = expandedSobel(yTrue)

    #calculate the sobel filters for yTrue and yPred
    #this generates twice the number of input channels 
    #a X and Y channel for each input channel
    sobelTrue = K.depthwise_conv2d(yTrue,filt,padding='same')
    sobelPred = K.depthwise_conv2d(yPred,filt,padding='same')

    #now you just apply the mse:
#     return K.mean(K.square(sobelTrue - sobelPred))
    return K.sum(K.abs(sobelPred - sobelTrue), axis=-1)


def sobelLoss3D(yTrue,yPred):

    #get the sobel filter repeated for each input channel
    filt = expandedSobel3D(yTrue)

    #calculate the sobel filters for yTrue and yPred
    #this generates twice the number of input channels 
    #a X and Y channel for each input channel
    sobelTrue = K.conv3d(yTrue, filt, padding='same')
    sobelPred = K.conv3d(yPred, filt, padding='same')

    #now you just apply the mse:
#     return K.mean(K.square(sobelTrue - sobelPred))
    return K.sum(K.abs(sobelPred - sobelTrue), axis=-1)
# def sobelLoss(yTrue, yPred):
# 
#     fx = K.variable([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#                        [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
#                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])
#     fy = K.variable([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
#                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                        [[1, 2, 1], [2, 4, 2], [1, 2, 1]]])
#     fz = K.variable([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                        [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
#                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])
# #     kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
# #                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
#     yTrue = K.tf.where(K.tf.is_nan(yTrue), K.tf.ones_like(yTrue) * 0, yTrue)
#     yPred = K.tf.where(K.tf.is_nan(yPred), K.tf.ones_like(yPred) * 0, yPred)
#     sobels = []
#     for k in [fx, fy, fz]:
#         kernels = K.expand_dims(k, -1)
#         kernels = K.expand_dims(kernels, -1)
#         sobel = K.conv3d(yTrue, kernels, padding='same')
#         sobels.append(sobel)
#     sum_sobel = K.square(sobels[0])+K.square(sobels[1])+K.square(sobels[2])
#     sobelTrue = K.sqrt(sum_sobel)
#     
#     sobels = []
#     for k in [fx, fy, fz]:
#         kernels = K.expand_dims(k, -1)
#         kernels = K.expand_dims(kernels, -1)
#         sobel = K.conv3d(yPred, kernels, padding='same')
#         sobels.append(sobel)
#     sum_sobel = K.square(sobels[0])+K.square(sobels[1])+K.square(sobels[2])
#     sobelPred = K.sqrt(sum_sobel)
#     
#     return K.sum(K.abs(sobelPred - sobelTrue), axis=-1)
