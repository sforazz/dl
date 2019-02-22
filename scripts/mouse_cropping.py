import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plot
from fpdf import FPDF
import nrrd
                                                                                                                                  
# 
# im = nib.load('/home/fsforazz/Desktop/CONT_16WK.nii.gz')
# header = im.get_header()
# pixDimX, pixDimY, pixDimZ = header.get_zooms()
# 
# im = im.get_data()
# 
# dimX, dimY, dimZ = im.shape                                                                                                                                             
# 
# sizeMMx = int(np.round(32/pixDimX)) ##here I'm assuming a mouse is 2cm wide and I add 1 cm margin, then I divide by the pix dim to find the number of voxels
#                                                                                                                                          
# im[im<-800] = -1024
# 
# for i in range(dimX):
#     if len(set(im[i, :, int(dimZ/2)])) != 1:
#         first_x = i
#         break 
# 
# for i in range(dimY): 
#     if len(set(im[:, i, int(dimZ/2)])) != 1:
#         first_y = i
#         break 
# 
# im_new = np.zeros((im.shape)) - np.abs(np.min(im))
# if first_x <= first_y:
#     if first_y < 100:
#         im_new[:, 200:, :]=im[:, 200:, :]
#     else:
#         im_new[:, first_y-10:, :]=im[:, first_y-10:, :]
# else:
#     print('MMMM are you sure??')
#     im_new[250:, :, :]=im[250:, :, :]                                                                                                                                      
# 
# m = []
# for sl in range(dimZ): 
#     th, im_th = cv2.threshold(im_new[:, :, sl], 0, 255, cv2.THRESH_BINARY_INV) 
#     im_floodfill = im_th.copy()
#     h, w= im_th.shape[:2] 
#     mask = np.zeros((h+2, w+2), np.uint8) 
#     res=cv2.floodFill(im_floodfill.astype(np.uint8), mask, (0,0), 255)  
#     mm = cv2.Canny(res[2][:dimX,:dimY], 0, 1) 
#     m.append(mm) 
# 
# m = np.asarray(m)
# m = np.swapaxes(m, 0, 2)                                                                                                                                                                           
# m = np.swapaxes(m, 0, 1)
# 
# deltaZ = int(np.ceil((dimZ-86)/2))
# mean_Z = int(np.ceil((dimZ)/2))
# x, y = np.where(m[:, :, mean_Z]==255)
# 
# indX = np.max(x) 
# indY = np.max(y)
# indZ = np.abs(deltaZ-(dimZ-deltaZ))
# 
# n_mice = int(np.round((np.max(x) - np.min(x))/sizeMMx))
# png4repo = []
# for i in range(n_mice):
#     new_image = np.zeros((86, 86, 86)) - np.abs(np.min(im))
#     if sizeMMx > 86:
#         deltaX = int((sizeMMx-86)/2)
#     else:
#         deltaX = 0
#     new_image[:sizeMMx,:,:indZ] = im_new[indX+deltaX-sizeMMx:indX-deltaX, indY-86:indY, deltaZ:dimZ-deltaZ]
#     fig, (ax1, ax2, ax3) = plot.subplots(1, 3)
#     fig.suptitle('test_2_mouse_{}'.format(i), fontsize=16)
#     im1 = ax1.imshow(np.rot90(new_image[int(sizeMMx/2), :, :]))
#     im2 = ax2.imshow(np.rot90(new_image[:, 40, :]))
#     im3 = ax3.imshow(np.rot90(new_image[:, :, int(indZ/2)], 1))
#     plot.savefig('/home/fsforazz/Desktop/test_2_mouse_{}.png'.format(i))
#     plot.close()
#     png4repo.append('/home/fsforazz/Desktop/test_2_mouse_{}.png'.format(i))
#     im2save = nib.Nifti1Image(new_image, affine=np.eye(4)) 
#     nib.save(im2save, '/home/fsforazz/Desktop/test_2_mouse_{}.nii'.format(i)) 
#     indX = indX-sizeMMx
# 
# pdf = FPDF('L', 'mm')
# pdf.add_page()
# for image in png4repo:
#     pdf.image(image)
# pdf.output("/home/fsforazz/Desktop/yourfile.pdf", "F")
# 
# print('Done!')

im, hd = nrrd.read('/home/fsforazz/Desktop/test_cheng/CONT_16WK_Sequence_2/TR4_CONT_16WK_CHENG.CT.SPEZIAL_CHENG_CT_MOUSE_(ERWACHSENER).0002.0001.2014.08.14.17.08.53.984375.167193997.nrrd')

dimX, dimY, dimZ = im.shape
# im[im<-800] = -1024
# mean_Z = int(dimZ/2)
# 
# for i in range(dimX):
#     if len(set(im[i, :, int(dimZ/2)])) != 1:
#         first_x = i
#         break 
# 
# for i in range(dimY): 
#     if len(set(im[:, i, int(dimZ/2)])) != 1:
#         first_y = i
#         break 
# 
# im_new = np.zeros((im.shape)) - np.abs(np.min(im))
# if first_x <= first_y:
#     if first_y < 100:
#         im_new[:, 200:, :]=im[:, 200:, :]
#     else:
#         im_new[:, first_y-10:, :]=im[:, first_y-10:, :]
# else:
#     print('MMMM are you sure??')
#     im_new[250:, :, :]=im[250:, :, :]                                                                                                                                      

# m = []
# for sl in range(dimZ): 
#     th, im_th = cv2.threshold(im_new[:, :, sl], 0, 255, cv2.THRESH_BINARY_INV) 
#     im_floodfill = im_th.copy()
#     h, w= im_th.shape[:2] 
#     mask = np.zeros((h+2, w+2), np.uint8) 
#     res=cv2.floodFill(im_floodfill.astype(np.uint8), mask, (0,0), 255)  
#     mm = cv2.Canny(res[2][:dimX,:dimY], 0, 1) 
#     m.append(mm) 
# 
# m = np.asarray(m)
# m = np.swapaxes(m, 0, 2)                                                                                                                                                                           
# m = np.swapaxes(m, 0, 1)
# 
deltaZ = int(np.ceil((dimZ-86)/2))
mean_Z = int(np.ceil((dimZ)/2))
# x,y= np.where(m[:,:,mean_Z]==255)
 
# indX = np.max(x) 
# indY = np.max(y)
indZ = np.abs(deltaZ-(dimZ-deltaZ))

im[im<-200] = -1024
x,y= np.where(im[:,:,mean_Z]!=-1024)
indY = np.max(y)
uniq = list(set(x))
xx = [uniq[0]]
for i in range(1, len(uniq)): 
    if uniq[i]!=uniq[i-1]+1:
        xx.append(uniq[i-1])
        xx.append(uniq[i])
        print(i)
xx.append(uniq[-1])
# xx = np.where(im[:, y[0], mean_Z]==255)[0]

n_mice = 0
png4repo = []
im, hd = nrrd.read('/home/fsforazz/Desktop/test_cheng/CONT_16WK_Sequence_2/TR4_CONT_16WK_CHENG.CT.SPEZIAL_CHENG_CT_MOUSE_(ERWACHSENER).0002.0001.2014.08.14.17.08.53.984375.167193997.nrrd')
# im = im.get_data()
for i in range(0, len(xx), 2):
    size = xx[i+1] - xx[i]
    mp = int((xx[i+1] + xx[i])/2)
    if size % 2 != 0:
        size = size+1
    new_image = np.zeros((86, 86, 86)) - np.abs(np.min(im))
    if size > 86:
        deltaX = int((size-86)/2)
    else:
        deltaX = 0
    sizeX=xx[i+1]-deltaX-(xx[i]+deltaX)
    new_image = im[mp-43:mp+43, indY-86:indY, deltaZ:dimZ-deltaZ]
#     new_image = im[xx[i]+deltaX:xx[i+1]-deltaX, indY-86:indY, deltaZ:dimZ-deltaZ]
    fig, (ax1, ax2, ax3) = plot.subplots(1, 3)
    fig.suptitle('test_2_mouse_{}'.format(i), fontsize=16)
    im1 = ax1.imshow(np.rot90(new_image[int(size/2), :, :]))
    im2 = ax2.imshow(np.rot90(new_image[:, 40, :]))
    im3 = ax3.imshow(np.rot90(new_image[:, :, int(indZ/2)], 1))
    plot.savefig('/home/fsforazz/Desktop/test_2_mouse_{}.png'.format(n_mice))
    plot.close()
    png4repo.append('/home/fsforazz/Desktop/test_2_mouse_{}.png'.format(n_mice))
    im2save = nib.Nifti1Image(new_image, affine=np.eye(4)) 
    nib.save(im2save, '/home/fsforazz/Desktop/test_2_mouse_{}.nii'.format(n_mice))
    n_mice += 1
#     indX = indX

pdf = FPDF('L', 'mm')
pdf.add_page()
for image in png4repo:
    pdf.image(image)
pdf.output("/home/fsforazz/Desktop/yourfile.pdf", "F")
 
print('Done!')