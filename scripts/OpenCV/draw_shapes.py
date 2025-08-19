# draw various shapes on black image


import numpy as np 
import cv2 


img = np.zeros((512, 512, 3), np.uint8) # black image  

"""

# draw a line
img = cv2.line( img ,  # image to draw on
    (0, 0), # starting point
    (256, 256),  # ending point
    (0, 255, 0), # color
    10 # thickness
) 

# draw a arrow
img = cv2.arrowedLine(img, (0, 256), (256, 256), (0, 0, 255), 3)


# draw a rectangle
img = cv2.rectangle(img, (384, 0), (256, 128), (0, 0, 255), -1)


img = cv2.circle(img, (447, 63), 63, (0, 255, 0), -1)

font = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.putText(img, 'OpenCV', (10, 100), font, 4, (255, 0, 255), 10, cv2.LINE_AA)
""" 
 
img = cv2.polylines(img, [np.array([[10, 5], [20, 30], [70, 20], [50, 10]])], True, (0, 255, 0), 3)

cv2.imshow('line', img ) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 
