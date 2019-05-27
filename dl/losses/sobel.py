import keras.backend as K


sobelFilter = K.variable([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], 
                           [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                          [[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], 
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[1, 2, 1], [2, 4, 2], [1, 2, 1]]],
                          [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                           [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]])
sobelFilter = K.expand_dims(sobelFilter, axis=-2)


def expandedSobel(inputTensor):

    #this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return sobelFilter * inputChannels

# def sobelLoss(yTrue,yPred):
# 
#     #get the sobel filter repeated for each input channel
#     filt = expandedSobel(yTrue)
# 
#     #calculate the sobel filters for yTrue and yPred
#     #this generates twice the number of input channels 
#     #a X and Y channel for each input channel
#     sobelTrue = K.conv3d(yTrue, filt, padding='same')
#     sobelPred = K.conv3d(yPred, filt, padding='same')
#     sobelTrue = K.sqrt(K.square(sobelTrue[:, :, :,:, 0])+K.square(sobelTrue[:, :, :,:, 1])+K.square(sobelTrue[:, :, :,:, 2]))
#     sobelPred = K.sqrt(K.square(sobelPred[:, :, :,:, 0])+K.square(sobelPred[:, :, :,:, 1])+K.square(sobelPred[:, :, :,:, 2]))
#     sobelPred = K.expand_dims(sobelPred, axis=-1)
#     sobelTrue = K.expand_dims(sobelTrue, axis=-1)
#     
# 
#     #now you just apply the mse:
#     return K.mean(K.square(sobelTrue - sobelPred))


def sobelLoss(yTrue, yPred):

    fx = K.variable([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                       [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])
    fy = K.variable([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                       [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                       [[1, 2, 1], [2, 4, 2], [1, 2, 1]]])
    fz = K.variable([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                       [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])
#     kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
    yTrue = K.tf.where(K.tf.is_nan(yTrue), K.tf.ones_like(yTrue) * 0, yTrue)
    yPred = K.tf.where(K.tf.is_nan(yPred), K.tf.ones_like(yPred) * 0, yPred)
    sobels = []
    for k in [fx, fy, fz]:
        kernels = K.expand_dims(k, -1)
        kernels = K.expand_dims(kernels, -1)
        sobel = K.conv3d(yTrue, kernels, padding='same')
        sobels.append(sobel)
    sum_sobel = K.square(sobels[0])+K.square(sobels[1])+K.square(sobels[2])
    sobelTrue = K.sqrt(sum_sobel)
    
    sobels = []
    for k in [fx, fy, fz]:
        kernels = K.expand_dims(k, -1)
        kernels = K.expand_dims(kernels, -1)
        sobel = K.conv3d(yPred, kernels, padding='same')
        sobels.append(sobel)
    sum_sobel = K.square(sobels[0])+K.square(sobels[1])+K.square(sobels[2])
    sobelPred = K.sqrt(sum_sobel)
    
    return K.sum(K.abs(sobelPred - sobelTrue), axis=-1)
