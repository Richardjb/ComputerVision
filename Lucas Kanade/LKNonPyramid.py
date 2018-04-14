import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import scipy.misc

class LKanade():
    def __init__(self, src, dst, windowSize):
        print ("Lucas Kanade initializing")
        # gets images and converts them to gray
        self.im1 = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        self.im2 = cv2.imread(dst, cv2.IMREAD_GRAYSCALE)

        # window size to be used with this LK method
        self.ww = windowSize
        self.w = self.ww/2
        # reduce size of image
        sc = 2;
        imHeight, imWidth = self.im1.shape
        nwHeight, nwWidth = np.round(imHeight/2),np.round(imWidth/2)
        self.im2c = cv2.resize(self.im2,(nwWidth,nwHeight))
        self.container = cv2.goodFeaturesToTrack(self.im2c,25,0.01,10)
        self.ptsOfInterest = []
        for i in range (0, len(self.container)):
            self.ptsOfInterest.append(self.container[i][0])

        print self.ptsOfInterest[1][0]
        print self.ptsOfInterest[1][1]
        print self.ptsOfInterest[1]
        self.C1 = cv2.goodFeaturesToTrack(self.im2c,25,0.01,10)
        self.C1 = self.ptsOfInterest
        self.C1 = self.C1 * sc

        self.C = []
        # changed from 1
        #k = 1
        k = 0
        for i in range (0, len(self.C1)):

            x_i = self.C1[i][0]
            y_i = self.C1[i][1]

            if (x_i - self.w >=1 and y_i-self.w >=1 and x_i+self.w<=len(self.im1[0]) -1 and y_i+self.w<=len(self.im1[i])):
                k = k+1
                self.C.append(self.C1[i])

        for i in self.container:
            x,y = i.ravel()
            x = int(x*2)
            y = int(y*2)
            cv2.circle(self.im1,(x,y), 3,0,-1)

        # 2x2 used to calc deriv
        kern1 = 0.25 * np.array(([-1, 1], [-1, 1]))
        kern2 = 0.25 * np.array(([-1, -1], [1, 1]))
        kern1 = np.fliplr(kern1)
        # actual derivative calculations over both frames
        fx = cv2.filter2D(self.im1, -1, kern1) + cv2.filter2D(self.im2, -1, kern1)
        fy = cv2.filter2D(self.im1, -1, kern2) + cv2.filter2D(self.im2, -1, kern2)
        ft = cv2.filter2D(self.im1, -1, 0.25 * np.ones((2, 2))) + \
        cv2.filter2D(self.im2, -1, -0.25 * np.ones((2, 2)))
        cv2.imshow("im2c", self.im1)
        cv2.imshow("im2", self.im2c)

        #create windows
        for k in range (0, len(self.C)):
            i = self.C[k][0]
            j = self.C[k][1]
            Ix = fx[i-self.w:i+self.w, j-self.w:j+self.w]
            Iy = fy[i-self.w:i+self.w, j-self.w:j+self.w]
            It = ft[i-self.w:i+self.w, j-self.w:j+self.w]

            Ix = Ix[:]
            Iy = Iy[:]
            b = -It[:] # to get b

            A = np.array([Ix,Iy])

            #nu = np.linalg.pinv(np.asarray(A)) * b
        denom = cv2.filter2D(fx**2, -1, self.w)*cv2.filter2D(fy**2, -1, self.w) - \
            cv2.filter2D((fx*fy), -1, self.w)**2
        #denom[denom == 0] = np.inf

        u = (-cv2.filter2D(fy**2, -1, self.w)*cv2.filter2D(fx*ft, -1, self.w) +
             cv2.filter2D(fx*fy, -1, self.w)*cv2.filter2D(fy*ft, -1, self.w)) / \
            denom
        v = (cv2.filter2D(fx*ft, -1, self.w)*cv2.filter2D(fx*fy, -1, self.w) -
             cv2.filter2D(fx**2, -1, self.w)*cv2.filter2D(fy*ft, -1, self.w)) / \
             denom

        # Create grid for display
        x = np.arange(0, self.im1.shape[1], 1)
        y = np.arange(0, self.im1.shape[0], 1)
        x, y = np.meshgrid(x, y)


        plt.figure()
        plt.imshow(self.im1, cmap='gray', interpolation='bicubic')
        plt.title("my lucas")


        step = 5
        plt.quiver(x[::step, ::step], y[::step, ::step],
                   u[::step, ::step], v[::step, ::step],
                   color='g', pivot='middle', headwidth=2, headlength=3)
        plt.show()


        cv2.waitKey(0)
