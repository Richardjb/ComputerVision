import cv2
import numpy as np
from matplotlib import pyplot as plt
import maxflow
import os

# tells np to print entire array
np.set_printoptions(threshold=np.nan)

class GraphCut:
    # directory of images ot be segmented
    graphCutDir = "GraphCutImages"

    # gets lists of files in the directories
    graphCutlst = os.listdir(graphCutDir)

    # used for user to seed data
    point1 = (0,0)
    point2 = (0,0)
    pointsPicked = 0

    # definitions for  on and off pixels
    ON_PX = 255
    OFF_PX = 0
    # used to prevent division by 0 and 0/0
    SIGMA = 1e-8

    def __init__(self):
        print("made facerec")
        # initializes values
        self.point1 = (0,0)
        self.point2 = (0,0)
        self.pointsPicked = 0

        # this loop does graph segmentation for all images in graph
        # cut foler
        for pic in self.graphCutlst:
            # builds full image reference to be loaded
            picName = self.graphCutDir + '/' + pic

            try:
                #img = np.array(Image.open(picName).convert('L'))
                # makes image global in the self namespace
                self.img = np.array(cv2.imread(picName, 0))
                self.colorImg = np.array(cv2.imread(picName))
                cv2.imshow(pic, self.img)
                # call back function, click event
                cv2.setMouseCallback(pic, self.ShowCoordinates)
                cv2.waitKey(0)
                print self.point1
                print self.point2
                cv2.destroyAllWindows()
            except:
                print("ERROR! " + pic + " File was not found! Trying next image\t")
                continue
            self.CalculateHistogram()
            self.CalculateUnaryWeights()
            self.CalculatePairWiseWeights()
            self.FlowAndSegmentation()


            self.pointsPicked = 0

    def FlowAndSegmentation(self):
        # Here is my energy field, which is defined as
        # unary multiplied by some constant lambda added with the
        # pairwise term
        constLam = 40
        Energy = constLam*self.Unary + self.PairWise

        # using the Energy function above, we can use
        # Pyflow to perform the Boykov/ Kologorov maxflow /mincut
        # algorithm

        # this performs the algorithm automatically when adding edges
        grph = maxflow.Graph[int](0,0)
        idents = grph.add_grid_nodes(self.img.shape)
        grph.add_grid_edges(idents, weights=Energy)
        grph.add_grid_tedges(idents, self.img, 255-self.img)
        # force praph to perform flowing algorithm
        grph.maxflow()
        output = grph.get_grid_segments(idents)

        new_Im = self.colorImg
        height, width = np.array(new_Im).shape[0], np.array(new_Im).shape[1]

        # shades background pixels red
        for j in range (height):
            for i in range (width):
                if (not output[j][i]):
                    new_Im[j][i] = (0,0,255) # RED

        # show resuting image
        cv2.imshow("Segmented Image", new_Im)
        cv2.waitKey(0)

    def CalculatePairWiseWeights(self):
        # Here is my I calculate my pairWise terms
        # pairwise terms are dependent on their neighboring terms
        height, width = self.img.shape
        H = np.zeros(self.img.shape)
        vecIm = H.reshape(self.img.size)
        padIm = np.pad(self.img,(1,), 'reflect').astype('float64')
        # curcumference calculation of pixels, radius of 1
        radius = 1
        circum = []
        for i in range (-radius, radius + 1):
            for j in range (-radius, radius + 1):
                circum.append([j,i])
        # calculate neighbors
        neighbors = []
        for j in range (1, height):
            for i in range (1, width):
                for y in range(-1,2):
                    for x in range (-1,2):
                        neighbors.append([j + y, i + x])
            pairSum = 0
            for w in range(len(circum)):
                point = [neighbors[w][0], neighbors[w][1]]
                pairSum += np.exp(-(np.square(padIm[y,x]-padIm[point[0],point[1]]))/2)
            index = (y-1)*height + (x-1)
            vecIm[index] = pairSum
        # This is the pairwise function that will be used in conjuction with the unary
        self.PairWise = np.reshape(vecIm, (height,width))
            #for w in range (2**3):



    def CalculateUnaryWeights(self):
        height, width = self.img.shape
        # this is my definition of my unary term
        imgVec = self.img.ravel()
        # using negative log likelihood similar to that listed on the readme
        unaryFunc = [-np.log(self.foreHist[index]/self.foreHist[index]) for index in imgVec]
        self.Unary = np.reshape(unaryFunc,(height,width))

    def CalculateHistogram(self):
        height, width = self.img.shape
        # creates foreGround Mat to help calc. FG histogram
        foregndPx = np.zeros(self.img.shape, np.uint8) # needs to be uint8 or calcHist won't work
        # creates backGround Mat to help calc. BG histogram
        backgndPx = np.ones(self.img.shape, np.uint8) # needs to be uint8 or calcHist won't work
        # marks area from Point1 to Point 2, seeds user data
        foregndPx[self.point1[0]:self.point2[0], self.point1[1]:self.point2[1]] = self.ON_PX
        backgndPx[self.point1[0]:self.point2[0], self.point1[1]:self.point2[1]] = self.OFF_PX
        # actual histogram calculation
        self.foreHist = cv2.calcHist([self.img], [0], foregndPx, [256], [0,256])
        self.backHist = cv2.calcHist([self.img], [0], backgndPx, [256], [0,256])
        # This is where we obtain likelihood function
        # The normalized hist. can give us a  nice probability
        # of whether a px is a background or foreground px
        # if px is not background it is foreground based on binary
        # unary function i.e. PR(FG) = 1 - PR(BG)
        self.foreHist = (self.foreHist + self.SIGMA)/(np.max(self.foreHist) + self.SIGMA)
        self.backHist = (self.backHist + self.SIGMA)/(np.max(self.backHist) + self.SIGMA)
        plt.figure(1)
        plt.title("BG intensities")
        plt.plot(self.backHist)
        plt.show()

        # change to show foreground intensities, they are constant in all cases
        if (False):
            plt.figure(2)
            plt.title("FG intensities")
            plt.plot(self.foreHist)
            plt.show()

    def ShowCoordinates(self, event, x, y, flags, userdata):

        if (event == cv2.EVENT_LBUTTONDOWN):
            try:
                if (self.pointsPicked == 0):
                    self.point1 = (x,y)
                    self.pointsPicked = 1
                    print ("x: %d\ty: %d" %(x,y))
                    return
            except:
                    print "Error assigning point"
            try:
                if (self.pointsPicked == 1):
                    self.point2 = (x, y)
                    self.pointsPicked = 2
                    print ("x: %d\ty: %d" %(x,y))
                    print ("Both Points Picked")
                    cv2.destroyAllWindows()
                    return
            except:
                print "Error assigning points"
