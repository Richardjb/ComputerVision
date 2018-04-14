import cv2
import numpy as np
from matplotlib import pyplot as plt

class LKanadePy():
    def __init__(self, src, dest, lvls, windowSize, guassianWidth):
        #initialize globals
        self.src = src
        self.dest = dest
        self.lvls = lvls
        self.windowSize = windowSize
        self.guassianWidth = guassianWidth
        self.MakePyramids()
        print ("Lucas Kanade Pyramid initializing")

    def MakePyramids(self):
        # create gaussinan py for src
        self.G = self.src.copy()
        gpA = [self.G]
        for i in range(self.lvls):
            self.G = cv2.pyrDown(self.G)
            gpA.append(self.G)

        # create gaussian py for dest
        self.G = self.dest.copy()
        gpB = [self.G]
        for i in range(self.lvls):
            self.G = cv2.pyrDown(self.G)
            gpB.append(self.G)

