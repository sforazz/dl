import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plot
from fpdf import FPDF
                                                                                                                                  

im = nib.load('/home/fsforazz/Desktop/test_2.nii.gz')
header = im.get_header()
pixDimX, pixDimY, pixDimZ = header.get_zooms()

im = im.get_data()

dimX, dimY, dimZ = im.shape                                                                                                                                             

sizeMMx = int(np.round(32/pixDimX)) ##here I'm assuming a mouse is 2cm wide and I add 1 cm margin, then I divide by the pix dim to find the number of voxels
                                                                                                                                         
im[im<-800] = -1024

for i in range(dimX):
    if len(set(im[i, :, int(dimZ/2)])) != 1:
        first_x = i
        break 

for i in range(dimY): 
    if len(set(im[:, i, int(dimZ/2)])) != 1:
        first_y = i
        break 

im_new = np.zeros((im.shape)) - np.abs(np.min(im))
if first_x <= first_y:
    im_new[:, 250:, :]=im[:, 250:, :]
else:
    print('MMMM are you sure??')
    im_new[250:, :, :]=im[250:, :, :]                                                                                                                                      

m = []
for sl in range(dimZ): 
    th, im_th = cv2.threshold(im_new[:, :, sl], 0, 255, cv2.THRESH_BINARY_INV) 
    im_floodfill = im_th.copy()
    h, w= im_th.shape[:2] 
    mask = np.zeros((h+2, w+2), np.uint8) 
    res=cv2.floodFill(im_floodfill.astype(np.uint8), mask, (0,0), 255)  
    mm = cv2.Canny(res[2][:dimX,:dimY], 0, 1) 
    m.append(mm) 

m = np.asarray(m)
m = np.swapaxes(m, 0, 2)                                                                                                                                                                           
m = np.swapaxes(m, 0, 1)

deltaZ = int(np.ceil((dimZ-86)/2))
x, y, z = np.where(m[:, :, deltaZ:dimZ-deltaZ]==255)

indX = np.max(x) 
indY = np.max(y)
indZ = np.abs(deltaZ-(dimZ-deltaZ))

n_mice = int(np.trunc((np.max(x) - np.min(x))/sizeMMx))
png4repo = []
for i in range(n_mice):
    new_image = np.zeros((86, 86, 86)) - np.abs(np.min(im)) 
    new_image[:sizeMMx,:,:indZ] = im_new[indX-sizeMMx:indX, indY-86:indY, deltaZ:dimZ-deltaZ]
    fig, (ax1, ax2, ax3) = plot.subplots(1, 3)
    fig.suptitle('test_2_mouse_{}'.format(i), fontsize=16)
    im1 = ax1.imshow(np.rot90(new_image[int(sizeMMx/2), :, :]))
    im2 = ax2.imshow(np.rot90(new_image[:, 40, :]))
    im3 = ax3.imshow(np.rot90(new_image[:, :, int(indZ/2)], 1))
    plot.savefig('/home/fsforazz/Desktop/test_2_mouse_{}.png'.format(i))
    plot.close()
    png4repo.append('/home/fsforazz/Desktop/test_2_mouse_{}.png'.format(i))
    im2save = nib.Nifti1Image(new_image, affine=np.eye(4)) 
    nib.save(im2save, '/home/fsforazz/Desktop/test_2_mouse_{}.nii'.format(i)) 
    indX = indX-sizeMMx

pdf = FPDF('L', 'mm')
pdf.add_page()
for image in png4repo:
    pdf.image(image)
pdf.output("/home/fsforazz/Desktop/yourfile.pdf", "F")

print('Done!')
