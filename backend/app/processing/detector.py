import numpy as np
import cv2 

# load in image and template in grayscale
img = cv2.imread('data/raw/tiktok_img.png', 0)
template = cv2.imread('assets/templates/tiktok_watermark.png', 0)


# get width and height of template (height, width)
w, h = template.shape

 
# All the 6 methods for comparison in a list
methods = [ cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

# test all methods
for method in methods:
    img2 = img.copy()

    result = cv2.matchTemplate(img2, template, method)

    # get the minimum and maximum values and their locations
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(min_loc, max_loc)

    # we want to take the min for these two methods, take max for the rest
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else: 
        location = max_loc


    bottom_right = (location[0] + w, location[1] + h)
    cv2.rectangle(img2, location, bottom_right, 255, 5)


    cv2.imshow('Match', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
# dimensions: (W - w + 1, H - h + 1) sliding around template over image 

