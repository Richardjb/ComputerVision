__author__ = 'Rich'
'''
import cv2
import numpy as np
from scipy import ndimage as ndImg

import matplotlib.pyplot as plt
import copy
from copy import deepcopy
from PIL import Image
'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import copy as cp
from scipy import ndimage
import cv2


class SIFT:
    def __init__(self, I, sigma, k, scales, octaves):

        # initialize variables
        # normal colored image
        self.I = I
        self.Sigma = sigma
        # grayscale image
        # self.Ig = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        self.Ig = Image.open("SIFT-input1.png").convert('L')
        # imaage of zeros filled with Ig intensities
        self.Izeros = np.zeros(np.array(self.Ig).shape)
        self.k = k
        self.scales = scales
        self.octaves = octaves

        self.Run()

    # performs Sift algorithm step by step
    def Run(self):
        # Blur original Image and making Pyramids
        self.MakePyramids()
        # create octa

    def MakePyramids(self):
        # self.Ibase = np.array(deepcopy(self.Ig))
        self.Izeros[:] = np.array(np.array(self.Ig))
        # Note: swapped because np.zeroes length width
        # self.Ipyramid = np.zeros((self.octaves,self.scales))
        self.Ipyramid = []


        # creating pyramids to calculate DOG later
        for i in range(0, self.octaves):
            # create a new level
            self.levels = []
            for j in range(0, self.scales):
                # self.sigma = (self.k)**((j-1))*1.6
                # self.hsize = np.int(np.ceil(7*self.sigma))
                # gausBoxSize = (2*self.hsize)+1
                # self.x = cv2.getGaussianKernel((2*self.hsize)+1,self.sigma)
                if (i < 1 and j < 1):
                    self.levels.append(cp.deepcopy(self.Izeros))
                elif (i > 0 and j < 1):
                    holder = cp.deepcopy(ndimage.zoom(self.Ipyramid[i - 1][0], 0.5, order=1))
                    # changeimage scales
                    self.levels.append(holder)
                self.Ipyramid.append(self.levels)

                # makes 1D gaussian 2d
                # self.H = self.makeGaussian(self.x)
                # self.Iblur = np.convolve(self.Ibase,self.H)
                # NOTE: combined steps create gaussian and convolve 2d array
                # self.IpyramidX = cv2.GaussianBlur(self.Ig, (gausBoxSize,gausBoxSize), self.sigma)
                # self.Ipyramid[j][i] = cv2.GaussianBlur(self.Ig, (gausBoxSize,gausBoxSize), self.sigma)
                # self.Ipyramid.append(cv2.GaussianBlur(self.Ibase, (gausBoxSize,gausBoxSize), self.sigma))
                # print self.Ipyramid

                # percentageIncrease = 1.5
                # curWidth, curHeight = self.Ibase.shape
                # self.Ibase = self.Ibase.resize((np.int(percentageIncrease * curWidth), np.int(percentageIncrease * curHeight)))
        # NOTE: may need to reverse input
        # xPy = np.asarray(self.Ipyramid)
        # xPy = np.array(self.Ipyramid)
        # print xPy.shape
        # print xPy[0]
        # cv2.resize(self.Ibase,self.Ibase,Size())

        # print xPy.shape
        # cv2.waitKey(0)
        self.octArr = []
        for i in range(0, self.octaves):
            self.levels = []

        for j in range(0, self.scales):
            if j == 0:
                temp = np.zeros(self.Ipyramid[i][0].shape)
                temp[:] = self.Ipyramid[i][0]

            sigma = np.power(self.k, j) * self.Sigma
            horsz = int(np.ceil(7 * sigma))
            horsz = 2 * horsz + 1

            tem = np.zeros(temp.shape)
            # apply the gaussian blue to the pyramid level
            tem[:] = cv2.GaussianBlur(temp, (horsz, horsz), sigma, sigma)

            self.levels.append(tem)

        self.octArr.append(self.levels)

    # creates 2d gaussian
    def makeGaussian(self, Gauss1d):
        """ Make a square gaussian kernel.

        Since the gaussian function is separable by multiplying
        2 1d kernals together you can get the 2d gaussian
        """
        x, y = Gauss1d.shape
        rotatedGuass = np.reshape(Gauss1d, (y, x))

        Gauss2d = np.dot(Gauss1d, rotatedGuass)

        return Gauss2d
