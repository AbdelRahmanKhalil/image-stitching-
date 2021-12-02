import numpy as np
from numpy.core.shape_base import atleast_1d
import cv2
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image


# img1_BGR = cv2.imread('image1.jpg')
# img1_rgb = cv2.cvtColor(img1_BGR, cv2.COLOR_BGR2RGB)

# Create point matrix get coordinates of mouse click on image


point_matrix = np.zeros((4, 2), np.int)

counter = 0


def mousePoints(event, x, y, flags, params):
    global counter
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN and counter < 4:
        point_matrix[counter] = x, y
        counter = counter + 1
        print("counter=", counter)


def get_points(image_path):
    # Read image
    img = cv2.imread(image_path)

    while counter < 5:
        for x in range(0, 4):
            cv2.circle(img, (point_matrix[x][0], point_matrix[x][1]), 3, (0, 255, 0), cv2.FILLED)

        if counter == 4:
            # starting_x = point_matrix[0][0]
            # starting_y = point_matrix[0][1]

            # ending_x = point_matrix[1][0]
            # ending_y = point_matrix[1][1]
            cv2.waitKey(10)
            break
            # Draw rectangle for area of interest
            # cv2.rectangle(img, (starting_x, starting_y), (ending_x, ending_y), (0, 255, 0), 3)

            # Cropping image
            # img_cropped = img[starting_y:ending_y, starting_x:ending_x]
            # cv2.imshow("ROI", img_cropped)

        # Showing original image
        cv2.imshow("Original Image ", img)
        # Mouse click event on original image
        cv2.setMouseCallback("Original Image ", mousePoints)
        # Printing updated point matrix

        # Refreshing window all time
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    print(point_matrix)


def create_a_matrix(point_matrix_11, point_matrix_22, crr_num):  # correspondence number
    a = np.zeros((2, 9), np.int)
    # a = [[ - point_matrix_11(crr_num,0) , - point_matrix_11(crr_num,1) , -1 , 0 , 0 , 0 , point_matrix_11(crr_num,0) * point_matrix_22(crr_num,0) , point_matrix_11(crr_num,1) * point_matrix_22(crr_num,0) , point_matrix_22(crr_num,0)],
    #      [ 0 , 0 , 0 , - point_matrix_11(crr_num,0) , - point_matrix_11(crr_num,1) , -1 , point_matrix_11(crr_num,0) * point_matrix_22(crr_num,1) , point_matrix_11(crr_num,1) * point_matrix_22(crr_num,1) , point_matrix_22(crr_num,1)]]

    a[0][0] = - point_matrix_11[crr_num][0]
    a[0][1] = - point_matrix_11[crr_num][1]
    a[0][2] = -1
    a[0][6] = point_matrix_11[crr_num][0] * point_matrix_22[crr_num][0]
    a[0][7] = point_matrix_11[crr_num][1] * point_matrix_22[crr_num][0]
    a[0][8] = point_matrix_22[crr_num][0]

    a[1][3] = - point_matrix_11[crr_num][0]
    a[1][4] = - point_matrix_11[crr_num][1]
    a[1][5] = -1
    a[1][6] = point_matrix_11[crr_num][0] * point_matrix_22[crr_num][1]
    a[1][7] = point_matrix_11[crr_num][1] * point_matrix_22[crr_num][1]
    a[1][8] = point_matrix_22[crr_num][1]
    return a


def compute_homography():
    a1 = create_a_matrix(point_matrix_1, point_matrix_2, 0)
    a2 = create_a_matrix(point_matrix_1, point_matrix_2, 1)
    a3 = create_a_matrix(point_matrix_1, point_matrix_2, 2)
    a4 = create_a_matrix(point_matrix_1, point_matrix_2, 3)

    a = np.concatenate((a1, a2, a3, a4), axis=0)
    print("a=", a)
    u, s, vh = np.linalg.svd(a)
    print("u=", u)
    print("s=", s)
    print("vh=", vh)
    print("vh shape=", np.shape(vh))
    h = vh[:, 8]
    print("h shape=", np.shape(h))
    print("h=", h)

    return h



def invWarping(warpedImage,H,originalImage):




def warpingImage(sourceImg, homography, destImg):
    warpedIMage = np.zeros((destImg.shape[0], destImg.shape[1], 3),dtype= int)

    for i in range(sourceImg.shape[0]):
        for j in range(sourceImg.shape[1]):
            point = np.reshape(np.array([i, j, 1]),(3,1))


            newPoints = np.dot(homography, point)
            x_dash = int(newPoints[0] / newPoints[2])
            y_dash = int(newPoints[1] / newPoints[2])
            if destImg.shape[0] > x_dash >= 0 and y_dash < destImg.shape[1] and y_dash >= 0:
                
                warpedIMage[x_dash][y_dash] = sourceImg[i][j]


    cv2.imshow("Warped image", warpedIMage.astype(np.uint8))
    cv2.waitKey(0)

if __name__ == "__main__":
    get_points("image1.jpg")
    point_matrix_1 = np.copy(point_matrix)
    counter = 0
    get_points("image2.jpg")
    point_matrix_2 = np.copy(point_matrix)
    H = compute_homography()

    homography = np.reshape(H, (3, 3))


    img1 = cv2.imread("image1.jpg")
    img2 = cv2.imread("image2.jpg")
    warpingImage(img2, H, img1)
