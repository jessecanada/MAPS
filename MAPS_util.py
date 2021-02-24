import os
import cv2
import numpy as np
import pandas as pd
from skimage import measure

def variance_of_laplacian(image):
    ''' Compute the Laplacian of the image and then return the focus
        measure, which is simply the variance of the Laplacian
        '''
    return cv2.Laplacian(image, cv2.CV_64F).var()

# find over-exposed areas
# returns the list of blobs, mask

# function to find blobs
def find_blobs(input_image):
    ''' Finds blobs (overexposed fluorescent signals),
        blurs the blobs.
        '''  
    # Gaussian blur
    blr = cv2.GaussianBlur(input_image,(5,5),cv2.BORDER_DEFAULT)

    # threshold
    th = cv2.threshold(blr, 235, 255, cv2.THRESH_BINARY)[1] # cv2.threshold gives 2 outputs, ret & th
                                                            # select the 2nd one
    # perform erosions and dilations to remove blobs of noise from image
    th = cv2.erode(th, None, iterations=2)
    th = cv2.dilate(th, None, iterations=4)
   
    # perform connected component analysis on thresholded image
    labels = measure.label(th, background=0, connectivity=2)
    
    # initialize a mask to store only the large components
    mask = np.zeros(th.shape, dtype='uint8')
    
    # initialize blobList to store a list of over-exposed cells found
    #blobList = [0]
    # loop over unique components
    for label in np.unique(labels):
        #ignore the background label
        if label ==0: #ignore background label
            continue
            
        # construct label mask for current label
        labelMask = np.zeros(th.shape, dtype='uint8')
        labelMask[labels == label] = 255
        
        # count the # of non-zero pixels 
        numPixels = cv2.countNonZero(labelMask)
    
        # if # of pixels in the component is greater than a threshold,
        # then add it to the overall mask
        if numPixels > 1000:
            mask = cv2.add(mask, labelMask)
            #blobList.append(label)
    return mask

def blur_blobs(input_image, input_mask):
    '''function to blur out blobs (aka over-expressed cells)'''
    blr = cv2.GaussianBlur(input_image, (101, 101), 0) # blur image
    dilatedMask = cv2.dilate(input_mask, None, iterations=25)
    output_image = np.where(dilatedMask, blr, input_image) # mask over original image with blurred
    return output_image

def create_folder(file_path, folder_name):
    '''creates a new folder'''
    if not os.path.exists(file_path + folder_name):
        print("Created folder '{}'.".format(folder_name))
        os.makedirs(file_path + folder_name)
    else:
        print("Folder '{}' already exists.".format(folder_name))
    
def get_names(folder_path):
    '''get unique file names'''
    ID_list = []
    for entry in os.scandir(folder_path):
        if entry.name.endswith('d0.TIF'):
            fName = entry.name
            index = fName.find('_')
            matchID = fName[index+1:-6]
            ID_list.append(matchID)
    return ID_list

def gamma_adj(image, gamma=1.0):
    ''' Build a lookup table mapping pixel values [0, 255] to
        their adjusted gamma values.
    '''
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def illum_corr(image):
    ''' performs illumination correction'''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100,100))
    topHat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return topHat

def hist_adj(img, clahe, offset):
    '''
        auto-adjust histogram
        '''
    hist, bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    #  get the cumulative distribution function
    cdf_normalized = cdf * hist.max()/ cdf.max()
    # find the slope of the cdf
    slope = np.diff(cdf_normalized)/np.diff(bins[:-1])
    # find the index of the first non-zero value of the derivative of the slope
    idx = np.nonzero(np.diff(slope))[0]
    # clahe
    if clahe == True:
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(5,5)) # set clipLimit to adj how much detail to bring up
        cl1 = clahe.apply(img)
        adjImg = cl1 + offset - idx[0]
    elif clahe == False:
        # slide the histogram over
        adjImg = img + offset - idx[0]
    return adjImg

def add_noise(input_img, noise_K):
  ''' add noise to input image
      noise_K: noise factor (0.1, 0.2...)
      '''
  noise =  np.random.normal(loc=0, scale=1, size=input_img.shape)
  noisy = np.clip((input_img.copy()/255.0 + noise*noise_K), 0, 1)
  #noisy = noisy*255.0
  return(noisy)

# function to draw rectangles as matplotlib.patches instance
def plot_rectangles(img, img_height, img_width, df_bboxes):
    """
    img: cv2 image
    img_height: height of image
    img_width: width of image
    df_bboxes will need the following features: x, y, w, h, prob
    return: matplotlib.patches instance
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    # Create figure and axes
    fig, ax = plt.subplots(1, 1)
    # Display the image
    ax.imshow(img[:,:,::-1]) # openCV loads img in bgr, this converts img to rgb for matplotlib
    # Create rectangle patches
    for i in range(len(df_bboxes)):
        x = df_bboxes.x[i]*img_width
        y = df_bboxes.y[i]*img_height
        w = df_bboxes.w[i]*img_width
        h = df_bboxes.h[i]*img_height
        rectangle = patches.Rectangle((x,y),w,h,linewidth=2,edgecolor='w',facecolor='none')
        ax.add_patch(rectangle)
    return fig, ax

def signaltonoise(a, axis=0, ddof=0):
    ''' a: numpy array
        axis: 0, 1 or None (over entire array)
        ddof: degree of freedom (default=0)
        '''
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def results_to_df(results, threshold):
    ''' Put Azure prediction results into a pandas DataFrame
        results: Azure Custom Vision prediction result instance
        threshold: float between 0.0 : 100.0
        '''
    x = []
    y = []
    w = []
    h = []
    prob = []

    for i, prediction in enumerate(results.predictions):
        if prediction.probability*100 > threshold:
            x.append(prediction.bounding_box.left)
            y.append(prediction.bounding_box.top)
            w.append(prediction.bounding_box.width)
            h.append(prediction.bounding_box.height)
            prob.append(prediction.probability*100)

    df_bbox = pd.DataFrame({'probability (%)': prob, \
                            'x': x, \
                            'y': y, \
                            'w': w, \
                            'h': h})
    return df_bbox