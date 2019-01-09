from skimage.filters.thresholding import threshold_otsu


def binarization(image):
    
    th = threshold_otsu(image)
    print(th)
    image[image>th] = 1
    image[image!=1] = 0
    
    return image