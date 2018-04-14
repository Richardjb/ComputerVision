import cv2
import numpy as np
import matplotlib

__author__ = 'richardjb'
import Image
import numpy as np
import matplotlib
import  cv2
import PIL
from HessCornerDetection import HessCD

class Program:

    def __init__(self):
        # filenames to be used with hess corner detection
        HessImg1 = "input1.png"
        HessImg2 = "input2.png"
        HessImg3 = "input3.png"

        # read images to be used with hess corner detection
        self.HessI1 = cv2.imread(HessImg1, 0)
        self.HessI2 = cv2.imread(HessImg2, 0)
        self.HessI3 = cv2.imread(HessImg3, 0)

    def Run(self):
        print("starting")
        # These are the best thresholds found by testing
        #HCD1 = HessCD(self.HessI1, 5, 550000)
        #HCD2 = HessCD(self.HessI2, 5, 70000)
        #HCD3 = HessCD(self.HessI3, 9, 3000000)
# initializes and runs program
def main():
    # creates instance of program class
    pg = Program()
    #
    pg.Run()

# main entry point of program
main()