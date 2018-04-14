import cv2
import numpy as np
import matplotlib

__author__ = 'richardjb'
import Image
import numpy as np
import matplotlib
import  cv2
from HessCornerDetection import HessCD
from HessCornerDetectionGauss import HessCDG
from HessCornerDetectionCornerness import HessCDC
from SIFT import SIFT
from Susan import Susan

class Program:

    def __init__(self):
        # filenames to be used with hess corner detection
        HessImg1 = "input1.png"
        HessImg2 = "input2.png"
        HessImg3 = "input3.png"
        # filenames to be used with SIFT
        siftImg1 = "SIFT-input1.png"
        siftImg2 = "SIFT-input2.png"
        #filenames to be used with Susan corner/edge detection
        self.susanImg1 = "susan_input1.png"
        self.susanImg2 = "susan_input2.png"

        # read images to be used with hess corner detection
        self.HessI1 = cv2.imread(HessImg1)
        self.HessI2 = cv2.imread(HessImg2)
        self.HessI3 = cv2.imread(HessImg3)
        # read images to be used with SIFT
        self.Sift1 = cv2.imread(siftImg1)
        self.Sift2 = cv2.imread(siftImg2)
        # read images to be used with SUSAN
        self.Susan1 = cv2.imread(self.susanImg1)
        self.Susan2 = cv2.imread(self.susanImg2)


    def Run(self):
        print("starting")
        # Hess Corner detection using 3 methods
        '''
        # These are the best thresholds found by testing
        print ("Performing Hessian corner Detection with lamda calculation")
        HCD1 = HessCD(self.HessI1, 5, 550000)
        HCD2 = HessCD(self.HessI2, 5, 70000)
        HCD3 = HessCD(self.HessI3, 9, 3000000)

        # I did not notice any change when playing with alpha
        print("Performing Hessian Corner Detection with gaussian")
        HCDG1 = HessCDG(self.HessI1, 5,  2800000, 1/25)
        #HCDG1 = HessCDG(self.HessI1, 5,  2800000, 1/100000000)
        HCDG2 = HessCDG(self.HessI2, 5, 10000, 1/25)
        HCDG3 = HessCDG(self.HessI3, 5, 10000, 1/50)
        print ("Performing Hessian corner Detection with Cornerness measure")
        HessCDC1 = HessCDC(self.HessI1, 5,  28000, 1/25)
        HessCDC1 = HessCDC(self.HessI2, 5,  28000, 1/25)
        HessCDC1 = HessCDC(self.HessI3, 5,  28000, 1/25)
        print("Complete!")
        '''
        # SIFT
        #siff = SIFT(self.Sift1, 1.6, 2**.5, 5, 4)

        # Bonus Susan
        susan = Susan(self.susanImg1)
        susan = Susan(self.susanImg2,process=True)

# initializes and runs program
def main():
    # creates instance of program class
    pg = Program()
    pg.Run()

# main entry point of program
main()