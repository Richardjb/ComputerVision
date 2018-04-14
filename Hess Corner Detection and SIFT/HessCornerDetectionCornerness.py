__author__ = 'Rich'
import cv2
import numpy as np
import scipy
import matplotlib
from copy import deepcopy


class HessCDC:
    '''
    HESSIAN MATRIX
    H_2(p,omega) = | Lxx(p,omega)  Lxy(p,omega)|
                   | Lxy(p,omega)  Lyy(p,omega)|
    '''

    def __init__(self, I, windowSize, threshold, alpha):

        # original colored image to place red dots
        self.coloredImage = I
        self.I = cv2.cvtColor(deepcopy(I), cv2.COLOR_BGR2GRAY)
        self.width, self.height = self.I.shape
        np.set_printoptions(threshold=np.nan)
        self.corners = []
        self.thresh = threshold
        # step 1
        # Computes gaussian blur in x and y directions
        self.sigma = 3
        self.maskSizeY = (5, 1)
        self.maskSizeX = (1, 5)

        self.alpha = alpha
        self.Lx = cv2.GaussianBlur(self.I, self.maskSizeX, self.sigma)
        self.Ly = cv2.GaussianBlur(self.I, self.maskSizeY, self.sigma)

        # Computes square derivatives
        self.Lxx = self.Lx * self.Lx
        self.Lxy = self.Lx * self.Ly
        self.Lyy = self.Ly * self.Ly
        # window (-m...m)x(-m...m)
        self.windowSize = windowSize
        # main process of discovering edges
        self.CornernessOperation()
        cv2.imshow("OrigIimage", self.coloredImage)

        for i in self.corners:
            # flip each point because x and y values were swapped
            self.coloredImage = cv2.circle(self.coloredImage, (i[1], i[0]), 1, (0, 0, 230), -1)

        cv2.setMouseCallback("OrigIimage", self.ShowCoordinates)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def CornernessOperation(self):
        offset = np.int(self.windowSize / 2)
        LxxSum = 0
        LxySum = 0
        LyySum = 0
        # loops through image
        for j in range(0, self.height):
            for i in range(0, self.width):
                # index of center pixel in window in respect to Image
                centerPixel = (i + offset, j + offset)
                # accounts for out of bounds errors
                if (centerPixel[0] + offset >= self.width or centerPixel[1] + offset >= self.height):
                    break
                # computes summation over window
                for u in range(centerPixel[0] - offset, centerPixel[0] + offset + 1):
                    for v in range(centerPixel[1] - offset, centerPixel[1] + offset + 1):
                        LxxSum += self.Lxx[u][v]
                        LxySum += self.Lxy[u][v]
                        LyySum += self.Lyy[u][v]
                        # print self.I[u][v]

                # finds cornerness measure
                # creates 2x2 matrix
                # arr = H_2(p,o)
                arr = [LxxSum, LxySum, LxySum, LyySum]
                curMatrix = np.reshape(arr, (2, 2))
                Det = np.linalg.det(curMatrix)
                Tr = np.matrix.trace(curMatrix)
                eigenValues, eigenVectors = np.linalg.eig(curMatrix)
                Cornerness = (eigenValues[0] * eigenValues[1]) - (self.alpha * (eigenValues[0] + eigenValues[1]))

                # Cornerness above threshold indicates a corner
                if (Cornerness > self.thresh):
                    self.corners.append(centerPixel)

                # resets values for next matrix computation
                LxxSum = 0
                LxySum = 0
                LyySum = 0

    # call back function that shows coordinates on mouse click
    def ShowCoordinates(self, event, x, y, flags, userdata):
        if (event == cv2.EVENT_LBUTTONDOWN):
            self.I = cv2.circle(self.coloredImage, (x, y), 3, (100, 100, 55), -1)
            print ("x: %d\ty: %d" % (x, y))
        cv2.imshow("OrigIimage", self.coloredImage)
