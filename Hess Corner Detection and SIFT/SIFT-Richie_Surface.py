__author__ = 'Rich'
import cv2

import numpy as np
import scipy

import matplotlib
from copy import deepcopy


class SIFT:

    def __init__(self, I, sigma, k, scales, octaves):

        # initialize variables
        self.I = I
        # grayscale image
        self.Ig = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        self.k = k
        self.scales = scales
        self.octaves = octaves

        self.Run()

    # performs Sift algorithm step by step
    def Run(self):
        # Blur original Image and making Pyramids
        self.MakePyramids()


    def MakePyramids(self):
        self.Ibase = np.array(deepcopy(self.Ig))
        # Note: swapped because np.zeroes length width
        #self.Ipyramid = np.zeros((self.octaves,self.scales))
        self.Ipyramid = []

        cv2.imshow("before gauss", self.Ig)
        # creating pyramids to calculate DOG
        for i in range (1, self.octaves + 1):
            for j in range (1, self.scales + 1):

                self.sigma = (self.k)**((j-1))*1.6
                self.hsize = np.int(np.ceil(7*self.sigma))
                gausBoxSize = (2*self.hsize)+1
                self.x = cv2.getGaussianKernel((2*self.hsize)+1,self.sigma)
                # makes 1D gaussian 2d
                #self.H = self.makeGaussian(self.x)
                #self.Iblur = np.convolve(self.Ibase,self.H)
                # NOTE: combined steps create gaussian and convolve 2d array
                #self.IpyramidX = cv2.GaussianBlur(self.Ig, (gausBoxSize,gausBoxSize), self.sigma)
                #self.Ipyramid[j][i] = cv2.GaussianBlur(self.Ig, (gausBoxSize,gausBoxSize), self.sigma)
                self.Ipyramid.append(cv2.GaussianBlur(self.Ibase, (gausBoxSize,gausBoxSize), self.sigma))
                #print self.Ipyramid

            percentageIncrease = 1.5
            curWidth, curHeight = self.Ibase.shape
            self.Ibase = self.Ibase.resize((np.int(percentageIncrease * curWidth), np.int(percentageIncrease * curHeight)))
        # NOTE: may need to reverse input
        #xPy = np.asarray(self.Ipyramid)
        xPy = np.array(self.Ipyramid)
        print xPy.shape
        print xPy[0]
        #cv2.resize(self.Ibase,self.Ibase,Size())

        print xPy.shape
        cv2.waitKey(0)
    # creates 2d gaussian
    def makeGaussian(self, Gauss1d):
        """ Make a square gaussian kernel.

        Since the gaussian function is separable by multiplying
        2 1d kernals together you can get the 2d gaussian
        """
        x,y = Gauss1d.shape
        rotatedGuass = np.reshape(Gauss1d, (y,x))

        Gauss2d = np.dot(Gauss1d, rotatedGuass)

        return Gauss2d


