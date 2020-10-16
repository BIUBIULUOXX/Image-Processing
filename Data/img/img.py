import numpy as np
import imageio
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage import restoration
import skimage.color as color 
import skimage.feature as feature
import skimage.transform as transform
import skimage.filters as filters 
import skimage.measure as measure

im = []

# read images from a file into a image object 
img_files = ['plantA_39.jpg','plantB_26.jpg','plantC_15.jpg','plantA_23.jpg','plantB_51.jpg','plantC_44.jpg']
for i in range(6):
    im.append(imageio.imread(img_files[i]))


# generate a greyscale version of each image
grayscale = []
for i in range(6):
    grayscale.append(rgb2gray(im[i]))

fig, axes = plt.subplots(2, 3, figsize=(8, 4))
ax = axes.ravel()
for i in range(6):
    ax[i].imshow(grayscale[i], cmap=plt.cm.gray),ax[i].axis('off')

ax[0].set_title('greyscale (A-39)')
ax[1].set_title('greyscale (B-26)')
ax[2].set_title('greyscale (C-15)')
ax[3].set_title('greyscale (A-23)')
ax[4].set_title('greyscale (B-51)')
ax[5].set_title('greyscale (C-44)')

plt.savefig( 'greyscale.png' )
fig.tight_layout()
plt.show()



# convert grayscale to binary image
from skimage.filters import threshold_otsu
thresh = []
binary = []
fig, axes = plt.subplots(2, 3, figsize=(8, 4))
ax = axes.ravel()

for i in range(6):
    thresh.append(threshold_otsu(grayscale[i]))
    binary.append(grayscale[i] > thresh[i])
    ax[i].imshow(binary[i], cmap=plt.cm.gray),ax[i].axis('off'),

ax[0].set_title('BW (A-39)')
ax[1].set_title('BW (B-26)')
ax[2].set_title('BW (C-15)')
ax[3].set_title('BW (A-23)')
ax[4].set_title('BW (B-51)')
ax[5].set_title('BW (C-44)')

plt.savefig( 'BW.png' )
plt.show()


# detect edges in each image

edges = []
fig, axes = plt.subplots(2, 3, figsize=(8, 4))
ax = axes.ravel()


for i in range(6):
    edges.append(feature.canny(grayscale[i], sigma=3))
    ax[i].imshow(edges[i],cmap = plt.cm.gray),ax[i].axis('off'),

    
ax[0].set_title('edges (A-39)')
ax[1].set_title('edges (B-26)')
ax[2].set_title('edges (C-15)')
ax[3].set_title('edges (A-23)')
ax[4].set_title('edges (B-51)')
ax[5].set_title('edges (C-44)')

plt.savefig( 'edges.png' )
plt.show()


# detect contours in each image

# find a good value for thresholding

threshold = []
contours = []

fig, axes = plt.subplots(2, 3, figsize=(8, 4))
ax = axes.ravel()

for i in range(6):
    threshold.append(filters.threshold_otsu(grayscale[i]) )
    # Find contours at threshold value found above
    contours.append(measure.find_contours(grayscale[i], threshold[i]))
    ax[i].imshow(grayscale[i], cmap=plt.cm.gray)
    for n, contour in enumerate(contours[i]):
        ax[i].plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax[i].axis('image'),ax[i].axis('off')

# Display the image and plot all contours found

ax[0].set_title('contours (A-39)')
ax[1].set_title('contours (B-26)')
ax[2].set_title('contours (C-15)')
ax[3].set_title('contours (A-23)')
ax[4].set_title('contours (B-51)')
ax[5].set_title('contours (C-44)')
plt.savefig( 'contours.png' )
plt.show()


# detect green 

fig, axes = plt.subplots(2, 3, figsize=(8, 4))
ax = axes.ravel()

for i in range(6):
    # convert to greyscale
    grayscale = rgb2gray(im[i])
    # convert to hsv
    hsv_img = rgb2hsv(im[i])
    
    # generate the white background
    img_255 = np.ones_like(im[0], np.uint8)* 255
    
    ## generate the filter 
    # green filter
    test0 = (hsv_img[:,:,0]>0)*(hsv_img[:,:,0]<1)
    test00 = np.dstack((test0,test0,test0))
    # threshold 
    thresh = threshold_otsu(grayscale)
    test1 = (grayscale<thresh)
    test11 = np.dstack((test1,test1,test1))
    
    # filter the background
    test_background = np.logical_not(test00*test11)
    
    # integrate the pic
    img_obj = im[i] * test00 * test11
    img_background = img_255*test_background
    imgg = img_obj + img_background
    ax[i].imshow(imgg)
    ax[i].axis('off')

    
ax[0].set_title('greens (A-39)')
ax[1].set_title('greens (B-26)')
ax[2].set_title('greens (C-15)')
ax[3].set_title('greens (A-23)')
ax[4].set_title('greens (B-51)')
ax[5].set_title('greens (C-44)')

plt.savefig( 'greens.png' )


# detect the straight lines in each image 

fig, axes = plt.subplots(2, 3, figsize=(8, 4))
ax = axes.ravel()
grayscale = []

for i in range(6):
    grayscale.append(rgb2gray(im[i]))
    # apply classic straight-line Hough transform on the Canny edges
    lines = transform.probabilistic_hough_line(edges[i], threshold = 10, line_length = 20,line_gap = 3 )
    # plot the lines
    for line in lines:
        p0, p1 = line 
        ax[i].plot((p0[0],p1[0]),(p0[1],p1[1])),
    ax[i].set_xlim(( 0, grayscale[i].shape[1] )),
    ax[i].set_ylim(( grayscale[i].shape[0], 0 )),    

# set titles for each images
ax[0].set_title('straight lines (A-39)')
ax[1].set_title('straight lines (B-26)')   
ax[2].set_title('straight lines (C-15)')   
ax[3].set_title('straight lines (A-23)')   
ax[4].set_title('straight lines (B-51)')       
ax[5].set_title('straight lines (C-44)')

fig.tight_layout()
plt.savefig( 'straightLine.png' )
plt.show()


