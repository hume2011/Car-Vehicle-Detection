import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog






# Read RGB image
def read_rgb(file_path):
    
    return mpimg.imread(file_path)

# Plot images
def display(img_list, size = (2,2), fig_size = (10,10), cm = None):
    fig, axes = plt.subplots(size[0], size[1], figsize=fig_size)
    axes = axes.ravel()
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for i in range(len(img_list)):
        axes[i].imshow(img_list[i], cmap=cm)
        axes[i].axis('off')
        #axes[i].set_title(file_path[i].split('/')[-1].split('.')[0])
        
        
# 3d Plots
def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


# Color(S layer) features
def get_s_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_layer = hsv[:,:,1]
    s_hist = np.histogram(s_layer, bins=32, range=(0,256))
    s_features = s_hist[0]

    return s_features


# HOG features
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True):
                         
    """
    Function accepts params and returns HOG features (optionally flattened) and an optional matrix for 
    visualization. Features will always be the first return (flattened if feature_vector= True).
    A visualization matrix will be the second return if visualize = True.
    """
    
    return_list = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys', transform_sqrt=False, 
                                  visualise= vis, feature_vector= feature_vec)
    
    
    return return_list
    
# Extract features from and image
def extract_features(img, orient=9, pix_per_cell=8, cell_per_block=2):
    s_features = get_s_features(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hog_features = get_hog_features(gray, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    
    return np.concatenate((s_features, hog_features))
    