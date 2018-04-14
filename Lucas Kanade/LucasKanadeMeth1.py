import cv2
import numpy as np
from matplotlib import pyplot as plt

class LucasKanadeMet1:

    def __init__(self, src, dest):
        print("beginning algorithm")

        self.src = src
        self.dest = dest

        # Reading the image
        #img1 = cv2.imread('teddy1.png', cv2.IMREAD_GRAYSCALE)
        #img2 = cv2.imread('teddy2.png', cv2.IMREAD_GRAYSCALE)

        img1 = cv2.cvtColor(self.src,cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(self.dest,cv2.COLOR_RGB2GRAY)

        # Convert the image to floating point for calculations
        img1 = np.float32(img1)
        img2 = np.float32(img2)

        # Calculate optical flow using two different algorithms
        u, v = self.lucas_kanade(img1, img2, np.ones((3, 3)))


        # Create grid for display
        x = np.arange(0, img1.shape[1], 1)
        y = np.arange(0, img1.shape[0], 1)
        x, y = np.meshgrid(x, y)

        # Display
        self.display_img(img1, 'My lucas_kanade')
        step = 2
        plt.quiver(x[::step, ::step], y[::step, ::step],
                   u[::step, ::step], v[::step, ::step],
                   color='r', pivot='middle', headwidth=2, headlength=2)

        plt.show()

    def display_img(self, img, name):
        plt.figure()
        plt.imshow(img, cmap='gray', interpolation='bicubic')
        plt.title(name)

    def get_derivatives(self, img1, img2):

        #affine 2x2 matrix
        kernel = np.array(([-1, 1], [-1, 1]))
        kernel2 = np.array(([-1, -1], [1, 1]))
        kernel = np.fliplr(kernel)

        # image x,y,t derivatives
        fx = cv2.filter2D(img1, -1, kernel) + cv2.filter2D(img2, -1, kernel)
        fy = cv2.filter2D(img1, -1, kernel2) + cv2.filter2D(img2, -1, kernel2)
        ft = cv2.filter2D(img1, -1, 0.25 * np.ones((2, 2))) + \
            cv2.filter2D(img2, -1, -0.25 * np.ones((2, 2)))

        return (fx, fy, ft)


    def lucas_kanade(self, img1, img2, window):
        """Lucase Kanade algorithm without pyramids.

        Implemented with convolution"""

        fx, fy, ft = self.get_derivatives(img1, img2)

        denom = cv2.filter2D(fx**2, -1, window)*cv2.filter2D(fy**2, -1, window) - \
            cv2.filter2D((fx*fy), -1, window)**2
        denom[denom == 0] = np.inf

        # pixel vectors
        u = (-cv2.filter2D(fy**2, -1, window)*cv2.filter2D(fx*ft, -1, window) +
             cv2.filter2D(fx*fy, -1, window)*cv2.filter2D(fy*ft, -1, window)) / \
            denom
        v = (cv2.filter2D(fx*ft, -1, window)*cv2.filter2D(fx*fy, -1, window) -
             cv2.filter2D(fx**2, -1, window)*cv2.filter2D(fy*ft, -1, window)) / \
            denom

        return (u, v)