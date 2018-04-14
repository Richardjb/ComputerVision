import cv2
import numpy as np
from copy import copy, deepcopy
from matplotlib import pyplot as plt
from KMeansClusteringGrayscale import KmeansGray
from KMeansClusteringColor import KmeansColor

def main():
    # read original images
    img1 = cv2.imread("input1.jpg")
    img2 = cv2.imread("input2.jpg")
    img3 = cv2.imread("input3.jpg")
    # create k-means object
    kmean = KmeansGray(img1,2)
    kmean2 = KmeansGray(img2,3)
    kmean3 = KmeansGray(img3,4)

    kmeanC = KmeansColor(img1,2)
    kmeanC2 = KmeansColor(img2,3)
    kmeanC3= KmeansColor(img3,4)



main()