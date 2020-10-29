def variance_of_laplacian(image):
# compute the Laplacian of the image and then return the focus
# measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

# find over-exposed areas
# returns the list of blobs, mask

# function to find blobs
def find_blobs(input_image):
    from skimage import measure
    # Gaussian blur
    blr = cv2.GaussianBlur(input_image,(5,5),cv2.BORDER_DEFAULT)

    # threshold
    th = cv2.threshold(blr, 235, 255, cv2.THRESH_BINARY)[1] # cv2.threshold gives 2 outputs, ret & th
                                                            # select the 2nd one
    # perform erosions and dilations to remove blobs of noise from image
    th = cv2.erode(th, None, iterations=2)
    th = cv2.dilate(th, None, iterations=4)
   
    # perform connected component analysis on thresholded image
    labels = measure.label(th, neighbors=8, background=0)
    
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

# function to blur out blobs (aka over-expressed cells)
def blur_blobs(input_image, input_mask):
    blr = cv2.GaussianBlur(input_image, (101, 101), 0) # blur image
    dilatedMask = cv2.dilate(mask, None, iterations=25)
    output_image = np.where(dilatedMask, blr, input_image) # mask over original image with blurred
    return output_image

def create_folder(file_path, folder_name):
    if not os.path.exists(file_path + folder_name):
        print("Created folder '{}'.".format(folder_name))
        os.makedirs(file_path + folder_name)
    else:
        print("Folder '{}' already exists.".format(folder_name))
    
# get unique file names
def get_names(folder_path):
    ID_list = []
    for entry in os.scandir(folder_path):
        if entry.name.endswith('d0.TIF'):
            fName = entry.name
            index = fName.find('_')
            matchID = fName[index+1:-6]
            ID_list.append(matchID)
    return ID_list

def gamma_adj(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def illum_corr(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100,100))
    topHat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return topHat
