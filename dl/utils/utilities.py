import time
from functools import wraps
import numpy as np
import scipy.signal as ss
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import tf_export
import tensorflow as tf

PROF_DATA = {}

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling

def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        print("Function %s called %d times. " % (fname, data[0])),
        print('Execution time max: %.3f, average: %.3f' % (max_time, avg_time))

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}
    

def sobel_3D(image, training=False):
    
    fx = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                   [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])
    fy = np.array([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                   [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                   [[1, 2, 1], [2, 4, 2], [1, 2, 1]]])
    fz = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                   [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                   [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])
    
    where_are_NaNs = np.isnan(image)
    image[where_are_NaNs] = 0
    if len(image.shape) > 3:
        if training:
            image = image[:, :, :, :, 0]
        sobel_3d = np.zeros((image.shape))
        for i in range(image.shape[0]):
            im_filt_x = ss.convolve(image[i, :, :, :, 0], fx, mode='same')
            im_filt_y = ss.convolve(image[i, :, :, :, 0], fy, mode='same')
            im_filt_z = ss.convolve(image[i, :, :, :, 0], fz, mode='same')
            sobel_3d[i, :] = np.expand_dims(np.sqrt((im_filt_x**2+im_filt_y**2+im_filt_z**2)), axis=-1)
    else:
        im_filt_x = ss.convolve(image, fx, mode='same')
        im_filt_y = ss.convolve(image, fy, mode='same')
        im_filt_z = ss.convolve(image, fz, mode='same')
        
        sobel_3d = np.sqrt((im_filt_x**2+im_filt_y**2+im_filt_z**2))
    
    return sobel_3d

# @tfplot.autowrap(figsize=(3, 3))
# def plot_imshow(img, *, fig, ax):
#     ax.imshow(img)


@tf_export('image.sobel_edges')
def sobel_edges(image):
    """Returns a tensor holding Sobel edge maps.
    
    Arguments:
      image: Image tensor with shape [batch_size, h, w, d] and type float32 or
      float64.  The image(s) must be 2x2 or larger.
    
    Returns:
      Tensor holding edge maps for each channel. Returns a tensor with shape
      [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
      [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
    """
    # Define vertical and horizontal Sobel filters.
    image = tf.where(tf.is_nan(image), tf.ones_like(image) * 0, image)
    fx = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                   [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])
    fy = np.array([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                   [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                   [[1, 2, 1], [2, 4, 2], [1, 2, 1]]])
    fz = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                   [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                   [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])
#     kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]

    sobels = []
    for k in [fx, fy, fz]:
        kernels = np.expand_dims(k, -1)
        kernels = np.expand_dims(kernels, -1)
        sobel = tf.nn.conv3d(image, kernels, strides = [1, 1, 1, 1, 1], padding='SAME')
        sobels.append(sobel)
    sum_sobel = tf.math.square(sobels[0])+tf.math.square(sobels[1])+tf.math.square(sobels[2])
    output = tf.math.sqrt(sum_sobel)
#     output = tf.concat([image, mag_sobel], axis=-1)
#     kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
#     kernels = np.expand_dims(kernels, -2)
#     kernels_tf = constant_op.constant(kernels, dtype=image.dtype)
#     
#     kernels_tf = array_ops.tile(kernels_tf, [1, 1, image_shape[-1], 1],
#                                 name='sobel_filters')
#     
#     # Use depth-wise convolution to calculate edge maps per channel.
#     pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
#     padded = array_ops.pad(image, pad_sizes, mode='REFLECT')
#     
#     # Output tensor has shape [batch_size, h, w, d * num_kernels].
#     strides = [1, 1, 1, 1]
#     output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')
#     
#     # Reshape to [batch_size, h, w, d, num_kernels].
#     shape = array_ops.concat([image_shape, [num_kernels]], 0)
#     output = array_ops.reshape(output, shape=shape)
#     output.set_shape(static_image_shape.concatenate([num_kernels]))
    return output

