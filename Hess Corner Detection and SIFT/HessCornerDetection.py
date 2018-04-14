__author__ = 'Rich'
import cv2
import numpy as np
import matplotlib
from copy import deepcopy
class HessCD:
    '''
    HESSIAN MATRIX
    H_1(p) = | Ixx(p)  Ixy(p)|
             | Ixy(p)  Iyy(p)|
    '''
    def __init__(self, I, windowSize,threshold):
        # original colored image to place red dots
        self.coloredImage = I
        # grayscale converted image for calculations
        self.I = cv2.cvtColor(deepcopy(I), cv2.COLOR_BGR2GRAY)
        self.width, self.height = self.I.shape
        np.set_printoptions(threshold=np.nan)
        self.corners = []
        self.thresh = threshold
        # step 1
        # Computes first order derivatives
        self.Ix = cv2.Sobel(self.I, cv2.CV_64F, 1,0)
        self.Iy = cv2.Sobel(self.I, cv2.CV_64F, 0,1)
        # Computes square derivatives
        self.Ixx = self.Ix * self.Ix
        self.Ixy = self.Ix * self.Iy
        self.Iyy = self.Iy * self.Iy
        #window (-m...m)x(-m...m)
        self.windowSize = windowSize
        # main process of discovering edges
        self.StructTensorOperation()
        cv2.imshow("OrigIimage", self.coloredImage)

        for i in self.corners:
            # flip each point because x and y values were swapped
            self.coloredImage = cv2.circle(self.coloredImage, (i[1],i[0]), 1, (0,0,200), -1)

        # call back function, click event
        cv2.setMouseCallback("OrigIimage", self.ShowCoordinates)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def StructTensorOperation(self):
        offset = np.int(self.windowSize / 2)
        ixxSum = 0
        ixySum = 0
        iyySum = 0
        # loops through image
        for j in range (0, self.height):
            for i in range(0, self.width):
                # index of center pixel in window in respect to Image
                centerPixel = (i+offset, j+offset)
                # accounts for out of bounds errors
                if (centerPixel[0] + offset >= self.width or centerPixel[1] + offset >= self.height):
                    break
                #computes summation over window
                for u in range (centerPixel[0] - offset, centerPixel[0] + offset + 1):
                    for v in range (centerPixel[1] - offset, centerPixel[1] + offset + 1):
                        ixxSum += self.Ixx[u][v]
                        ixySum += self.Ixy[u][v]
                        iyySum += self.Iyy[u][v]
                        #print self.I[u][v]

                # finds eigen values and matrix for current calculated Matrix
                # creates 2x2 matrix
                arr = [ixxSum,ixySum,ixySum,iyySum]
                curMatrix = np.reshape(arr, (2,2))
                eigenValues, eigenVectors = np.linalg.eig(curMatrix)

                # two large eigen values indicates a corner
                if (eigenValues[0] > self.thresh and eigenValues[1] > self.thresh):
                    self.corners.append(centerPixel)

                #resets values for next matrix computation
                ixxSum = 0
                ixySum = 0
                iyySum = 0



    # call back function that shows coordinates on mouse click
    def ShowCoordinates(self, event, x, y, flags, userdata):
        if (event == cv2.EVENT_LBUTTONDOWN):
            self.coloredImage = cv2.circle(self.coloredImage,(x,y), 3, (100,0,0), -1)
            print ("x: %d\ty: %d" %(x,y))
        cv2.imshow("OrigIimage", self.coloredImage)


