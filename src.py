import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
# img1_BGR = cv2.imread('image1.jpg')
# img1_rgb = cv2.cvtColor(img1_BGR, cv2.COLOR_BGR2RGB)

# Create point matrix get coordinates of mouse click on image
point_matrix = np.zeros((2,2),np.int)
 
counter = 0
def mousePoints(event,x,y,flags,params):
    global counter
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN and counter < 2:
        point_matrix[counter] = x,y
        counter = counter + 1
        print("counter=", counter)

def get_points(image_path): 
    # Read image
    img = cv2.imread(image_path)
    
    while counter < 3:
        for x in range (0,2):
            cv2.circle(img,(point_matrix[x][0],point_matrix[x][1]),3,(0,255,0),cv2.FILLED)
    
        if counter == 2:
            starting_x = point_matrix[0][0]
            starting_y = point_matrix[0][1]
    
            ending_x = point_matrix[1][0]
            ending_y = point_matrix[1][1]
            cv2.waitKey(10)
            break
            # Draw rectangle for area of interest
            #cv2.rectangle(img, (starting_x, starting_y), (ending_x, ending_y), (0, 255, 0), 3)
    
            # Cropping image
            #img_cropped = img[starting_y:ending_y, starting_x:ending_x]
            #cv2.imshow("ROI", img_cropped)
    
        # Showing original image
        cv2.imshow("Original Image ", img)
        # Mouse click event on original image
        cv2.setMouseCallback("Original Image ", mousePoints)
        # Printing updated point matrix
        
        # Refreshing window all time
        cv2.waitKey(1)
    print(point_matrix)

get_points("image1.jpg") 
point_matrix_1 =  point_matrix
counter = 0
get_points("image2.jpg")
point_matrix_2 =  point_matrix