__author__ = 'Rich'

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from copy import deepcopy as dp

import cv2


class Susan:
    def __init__(self, imgName, maskRad=3, process=False):
        # image processed?
        self.process = process
        # image read in as grey scale and copy for calc.
        try:
            I = cv2.imread(imgName, 0)
        except:
            print(imgName + " was not found! Please place in root directory of program")
            return
        # converts image to np.array, NOTE: possibly redundant cv2 reads image as np.array
        self.I = np.array(I)
        self.Icolor = np.array(cv2.imread(imgName))
        self.Izeros = np.zeros(I.shape)
        self.maskRad = maskRad

        # handles noise removal, sharpening and blurring of image
        # NOTE: after processing noisy image, the points become really similar to
        # the points of the non noisy image therefore it is very successful
        if (self.process):
            self.Preprocess()
        else:
            self.Izeros[:] = self.I

        # computes USAN for every pixel using given mask size
        self.ComputeUSAN()
        # Apply non-max suppression to find corners
        self.NonMaxSuppression()
        # Plot points and show
        self.PlotPoints()

    def PlotPoints(self):
        cv2.imshow("Non-Plotted image", self.Icolor)
        # plots all of the corners in the image
        for i in self.valPoints:
            self.Icolor = cv2.circle(self.I, (i[1],i[0]), 1, (100,0,250), -1)

        cv2.imshow("Plotted image", self.Icolor)
        cv2.waitKey(0)

    def NonMaxSuppression(self):
        # 3.2 non max suppression
        windowSize = self.maskRad * 2 - 1
        xLowLim = int(np.floor(windowSize / 2))
        xUpperLim = int(self.Izeros.shape[0] - np.floor(windowSize / 2))
        yLowLim = int(np.floor(windowSize / 2))
        yUpperLim = int(self.Izeros.shape[1] - np.floor(windowSize / 2))
        rad = int(np.floor(windowSize / 2))

        # x,y coordinates of corners/edges
        xCoor = []
        yCoor = []
        mask1 = np.zeros((windowSize, windowSize))
        for i in range(xLowLim, xUpperLim):
            for j in range(yLowLim, yUpperLim):

                if self.R[i][j] == 0:
                    continue
                mask1[:] = self.R[i - rad:i + rad + 1, j - rad:j + rad + 1]
                loc = np.argmax(mask1)
                if loc == 12:
                    xCoor.append(i)
                    yCoor.append(j)

        # valuable susan points
        self.valPoints = zip(xCoor,yCoor)
        if (self.process):
            print ("Preprocessed SUSAN ALG. points")
        else:
            print ("SUSAN ALG. points")
        print self.valPoints

    def ComputeUSAN(self):
        # (3.2 in HW pdf)
        height, width = self.Izeros.shape
        self.R = np.zeros(self.Izeros.shape)
        maskRad = 3
        # beginning value unInit
        nMax = -1
        totLen = 2 * maskRad + 1
        # threshold for image
        # NOTE: 20 is about the optimum value for thresholding
        # any higher eliminates points any lower adds excess points
        t = 20

        xLowLim = maskRad
        xUpperLim = height - maskRad
        yLowLim = maskRad
        yUpperLim = width - maskRad

        mask = np.zeros((totLen, totLen))
        # loop through image using circular mask
        for i in range(xLowLim, xUpperLim):
            for j in range(yLowLim, yUpperLim):
                # NOTE: ':' is shortcut for writing for loops with matrices
                mask[:] = self.Izeros[i - self.maskRad:i + self.maskRad + 1, j - self.maskRad:j + self.maskRad + 1]

                # the actual USAN calculation formula
                len = 2 * self.maskRad + 1
                # distance to pixel r
                dist = maskRad
                # center brightness r0
                nucleus = mask[dist][dist]

                # loops over mask's window
                for x in range(0, len):
                    for y in range(0, len):
                        # 3.1 formula
                        if ((x - dist) * (x - dist) + (y - dist) * (y - dist) <= self.maskRad * self.maskRad):
                            mask[x][y] = np.exp(-np.power(((mask[x][y] - nucleus) / t), 6))
                        else:
                            mask[x][y] = 0

                # make nucleus 0 for summation of circum. pixels
                mask[maskRad][maskRad] = 0
                # sum up mask
                n = np.sum(mask)

                self.R[i][j] = n
                if nMax < n:
                    nMax = n


        # threshold R matrix
        g = nMax / 2
        for i in range(xLowLim, xUpperLim):
            for j in range(yLowLim, yUpperLim):
                if (self.R[i][j] >= g):
                    self.R[i][j] = 0
                else:
                    self.R[i][j] = g - self.R[i][j]


    def Preprocess(self):
        # removes noise from image
        self.IzerosNoiseRemoval = np.zeros(self.I.shape)
        self.IzerosNoiseRemoval[:] = cv2.fastNlMeansDenoising(self.I, None, 40, 7, 21)

        imBlur = ndimage.gaussian_filter(self.IzerosNoiseRemoval, 3)
        imBlurFilt = ndimage.gaussian_filter(imBlur, 1)

        # the smaller sensitivity is the closer to having just corner points we get
        # the larger it is the closer we are to tracing edges
        sensitivity = .0001
        self.imSharp = imBlur + (sensitivity * (imBlur - imBlurFilt))
        # copies values into zeros container
        self.Izeros[:] = self.imSharp
