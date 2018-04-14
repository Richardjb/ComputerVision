import cv2
import numpy as np
from numpy import multiply
from matplotlib import pyplot as plt

class MyLK:

    def __init__(self, src, dest):
        print("beginning algorithm")
        # Reading the image
        self.img1 = cv2.imread('teddy1.png', cv2.IMREAD_GRAYSCALE)
        self.img2 = cv2.imread('teddy2.png', cv2.IMREAD_GRAYSCALE)
        self.origIm = cv2.imread('teddy1.png')
        self.origIm2 = cv2.imread('teddy2.png')


        self.GetGoodFeatures()
        #self.ShowGoodFeatures()

        # Convert the image to floating point for calculations
        #img1 = np.float32(img1)
        #img2 = np.float32(img2)

        # Calculate optical flow using two different algorithms
        u, v = self.LucasKanade2(self.img1, self.img2, np.ones((3, 3)))
        return
        # Create grid for display
        x = np.arange(0, img1.shape[1], 1)
        y = np.arange(0, img1.shape[0], 1)
        x, y = np.meshgrid(x, y)

        # Display
        self.display_img(img1, 'My lucas_kanade')
        step = 5
        plt.quiver(x[::step, ::step], y[::step, ::step],
                   u[::step, ::step], v[::step, ::step],
                   color='g', pivot='middle', headwidth=2, headlength=3)

        plt.show()

    def display_img(self, img, name):
        plt.figure()
        plt.imshow(img, cmap='gray', interpolation='bicubic')
        plt.title(name)

    # gets good features form both images
    def GetGoodFeatures(self):
        # use cv2 to find corners
        self.GFTT1 =  cv2.goodFeaturesToTrack(self.img1,25,0.01,10)
        self.GFTT2 =  cv2.goodFeaturesToTrack(self.img2,25,0.01,10)

        # array to hold valid points
        self.C = []
        # change shape of features
        for i in range (0, len(self.GFTT1)):
            self.C.append(self.GFTT1[i])

        print("empty")
    # shows good features in both images
    def ShowGoodFeatures(self):
        # plotting points on image 1
        for i in self.GFTT1:
            x,y = i.ravel()
            cv2.circle(self.origIm, (x,y), 3,255,-1)
        # plotting points on image 2
        for i in self.GFTT2:
            x,y = i.ravel()
            cv2.circle(self.origIm2, (x,y), 3,255,-1)

        # finally displaying images
        cv2.imshow("im1", self.origIm)
        cv2.imshow("im2", self.origIm2)
        cv2.waitKey(0)

    def DeriveImages(self, img1, img2):

        # use 2x2 for quick calculation
        kern = 0.25 * np.array(([-1, 1], [-1, 1]))
        kern2 = 0.25 * np.array(([-1, -1], [1, 1]))
        flippedKern = np.fliplr(kern)

        # NOTE: method derived from http://www.cs.ucf.edu/~gvaca/REU2013/p4_opticalFlow.pdf
        fx = cv2.filter2D(img1, -1, flippedKern) + cv2.filter2D(img2, -1, flippedKern)
        fy = cv2.filter2D(img1, -1, kern2) + cv2.filter2D(img2, -1, kern2)
        ft = cv2.filter2D(img1, -1, 0.25 * np.ones((2, 2))) + \
            cv2.filter2D(img2, -1, -0.25 * np.ones((2, 2)))

        return (fx, fy, ft)


    def LucasKanade(self, img1, img2, window):
        """Lucase Kanade algorithm without pyramids.

        Implemented with convolution"""

        fx, fy, ft = self.get_derivatives(img1, img2)

        denom = cv2.filter2D(fx**2, -1, window)*cv2.filter2D(fy**2, -1, window) - \
            cv2.filter2D((fx*fy), -1, window)**2
        denom[denom == 0] = np.inf

        u = (-cv2.filter2D(fy**2, -1, window)*cv2.filter2D(fx*ft, -1, window) +
             cv2.filter2D(fx*fy, -1, window)*cv2.filter2D(fy*ft, -1, window)) / \
            denom
        v = (cv2.filter2D(fx*ft, -1, window)*cv2.filter2D(fx*fy, -1, window) -
             cv2.filter2D(fx**2, -1, window)*cv2.filter2D(fy*ft, -1, window)) / \
            denom

        return (u, v)

    def LucasKanade2(self, img1, img2, window):
        """Lucase Kanade algorithm without pyramids.

        Implemented with convolution"""
        self.w = 20
        [fx, fy, ft] = self.DeriveImages(img1, img2)
        # creates 1d array for u,v calc
        u = np.zeros((len(img1),1))
        v = np.zeros((len(img1),1))
        # TODO: change to pass in size
        windowSize = 40
        halfWindow = np.floor(40/2)

                #create windows
        for k in range (0, len(self.GFTT1)):
            i = self.C[k][0][0]
            j = self.C[k][0][1]
            Ix = fx[i-self.w:i+self.w, j-self.w:j+self.w]
            Iy = fy[i-self.w:i+self.w, j-self.w:j+self.w]
            It = ft[i-self.w:i+self.w, j-self.w:j+self.w]


            #Ix = Ix.transpose()
            #Iy = Iy.transpose()
            #b = -It.transpose() # to get b

            Ix = Ix[:]
            Iy = Iy[:]
            b = -It[:] # to get b

            A = np.array([Ix,Iy])
            print A
            nu = np.dot(np.linalg.pinv(A.all(0)) , b)
            return (3,4)

            U = mulRes
            print U
            return (3,2)