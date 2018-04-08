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


# Binned color features
def bin_spatial(img, size=(32, 32)):
    
    features = cv2.resize(img, size).ravel() 
    
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    
    return hist_features

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
    
# Extract features from an image
def extract_features(img, orient=9, pix_per_cell=8, cell_per_block=2):
    bin_features = bin_spatial(img, size=(32,32))
    hist_features = color_hist(img)
    s_features = get_s_features(img)
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ch1 = yuv[:,:,0]
    ch2 = yuv[:,:,1]
    ch3 = yuv[:,:,2]
    hog_features1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    hog_features2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    hog_features3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    
    return np.concatenate((bin_features, hist_features, s_features, hog_features1, hog_features2, hog_features3))


# Generate windows for vehicle searching 
def slide_window(img, x_range=[None,None], y_range=[None,None], wind_size=(64,64), overlap=(0.5,0.5)):
    
    if x_range[0] == None:
        x_range[0] = 0
    if x_range[1] == None:
        x_range[1] = img.shape[1]
    if y_range[0] == None:
        y_range[0] = 0
    if y_range[1] == None:
        y_range[1] = img.shape[0]
        
    ## Compute the span of x and y 
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    
    ## Compute the buffer of x and y
    x_buffer = np.int(wind_size[0]*overlap[0])
    y_buffer = np.int(wind_size[1]*overlap[1])
            
    ## Compute the pixels for one step
    x_step = np.int(wind_size[0]*(1 - overlap[0]))
    y_step = np.int(wind_size[1]*(1 - overlap[1]))
         
    ## Compute the number of windows
    xn_windows = np.int((x_span-x_buffer)/x_step)
    yn_windows = np.int((y_span-y_buffer)/y_step)
    
    ## generate windows
    window_list = []
    for ys in range(yn_windows):
        for xs in range(xn_windows):
            x_start = xs*x_step + x_range[0]
            x_end = x_start + wind_size[0]
            y_start = ys*y_step + y_range[0]
            y_end = y_start + wind_size[1]
            window_list.append(((x_start,y_start),(x_end,y_end)))
    return window_list


# Classify windows
def classify_windows(img, windows, scaler, classifier):
    imcopy = np.copy(img)
    car_windows = []
    for wind in windows:
        img_toclassify = imcopy[wind[0][1]:wind[1][1],wind[0][0]:wind[1][0],:]
        img_toclassify = cv2.resize(img_toclassify,(64,64))
        features = extract_features(img_toclassify, orient=9, pix_per_cell=8, cell_per_block=2)
        scaled_features = scaler.transform(features.reshape(1,-1))
        pred = classifier.predict(scaled_features)
        if pred == 1:
            car_windows.append(wind)
    return car_windows


# Draw windows
def draw_windows(img, windows, color=(0,255,0), thick=2):
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for window in windows:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, window[0], window[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
'''
# Find cars
def find_cars(img, y_start, y_stop, cell_per_step=2, orient=9, pix_per_cell=8, cell_per_block=2, scale=1):
    imcopy = np.copy(img)
    im_tosearch = imcopy[y_start:y_stop,:,:]
    gray_tosearch = cv2.cvtColor(im_tosearch, cv2.COLO_RGB2GRAY)
    
    if scale!=1:
        gray_tosearch = cv2.resize(gray_tosearch,
                                   (gray_tosearch.shape[1]/scale,gray_tosearch.shape[1]/scale))
    hog_features = get_hog_features(gray_tosearch, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    
    window = 64
    
    ## Compute the number of cells in x and y
    nx_cells = gray_tosearch.shape[1]//pix_per_cell
    ny_cells = gray_tosearch.shape[0]//pix_per_cell
    
    ##Compute the nunmber of cells in a window
    nwind_cells = window//pix_per_cell
'''  


# Generate heatmap
def add_heat(heatmap, bbox_list):
   
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap


# Filter heatmap
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


# Draw labeled windows
def draw_labeled_windows(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
        cv2.putText(img, 'car', bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    # Return the image
    return img


# Process one image
def process_image(img, scaler, classifier):
    
    windows1 = utils.slide_window(img,x_range=[None,None],y_range=[360,656],wind_size=(256,256),overlap=ol)
    windows2 = utils.slide_window(img,x_range=[None,None],y_range=[360,656],wind_size=(192,192),overlap=ol)
    windows3 = utils.slide_window(img,x_range=[None,None],y_range=[360,656],wind_size=(128,128),overlap=ol)
    windows4 = utils.slide_window(img,x_range=[None,None],y_range=[360,656],wind_size=(64,64),overlap=ol)
    #windows5 = utils.slide_window(test_image,x_range=[None,None],y_range=[360,432],wind_size=(64,64),overlap=ol)
    #windows6 = utils.slide_window(test_image,x_range=[None,None],y_range=[360,452],wind_size=(64,64),overlap=ol)
    windows = windows1 + windows2 + windows3 + windows4
    car_windows = utils.classify_windows(img, windows, scaler, clf)

    heat = np.zeros_like(img[:,:,0])
    heat_img = utils.add_heat(heat, car_windows)
    
    thres_img = utils.apply_threshold(heat_img, 6)
    
    labels = label(thres_img)
    
    result_img = utils.draw_labeled_windows(img,labels)
    
    return result_img