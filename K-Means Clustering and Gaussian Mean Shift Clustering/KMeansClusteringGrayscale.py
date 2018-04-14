import cv2
import numpy as np
from copy import copy, deepcopy
from matplotlib import pyplot as plt
import random


class KmeansGray:
    def __init__(self, img, K):
        self.origI = img
        # K = # of clusters
        self.K = K
        # I = image in grayscale
        self.I = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        # centroids = the mean of each cluster, there are k centroids
        self.centroids  = []
        # create cluster array
        self.cluster = []
        # note the index of each cluster array matches the number
        # of clusters
        self.PerformAlg()
        # calculate efficiency of algorithm
        #self.AlgorithmAnalysis()
        #show results
        self.DisplayClusteredImage()


    def PickRandomCentroids(self):
        # the algorithm must be initialized with k random intensities
        # from the dataset
        self.h,self.w = self.I.shape
        #for i in range (0, self.K):
            #self.centroids.append(0)
        # pick random locations and set as centroids
        # the intensities will be used not distance
        for i in range (self.K):
            x = random.randrange(0, self.w)
            y = random.randrange(0, self.h)
            intensity = self.I[y][x]
            self.centroids.append((x,y))
            #self.centroids[i] = intensity

    def FillClusters(self):

        # assign all points in image to cluster
        for j in range(self.h):
            for i in range(self.w):
                pointIntensity = self.I[j][i]
                centroidStartIntensity = self.I[self.centroids[0][1]][self.centroids[0][0]]
                clusterToBeAssignedTo = 0
                minDistance = np.abs(pointIntensity - centroidStartIntensity)
                # compare with distance to all centroids
                for centroidIndex in range(len(self.centroids)):
                    compareValue = np.abs(pointIntensity - self.I[self.centroids[centroidIndex][1]][self.centroids[centroidIndex][0]])
                    if ( compareValue < minDistance):
                        minDistance = compareValue
                        clusterToBeAssignedTo = centroidIndex
                # final cluster assignment
                self.cluster[clusterToBeAssignedTo].append((i,j))


    def PerformAlg(self):
        # initializes with random intensities
        self.PickRandomCentroids()
        # creates centroids in form (x,y)
        for i in range(len(self.centroids)):
            self.cluster.append([])
        #used to deallocate points from clusters
        resetCluster = deepcopy(self.cluster)

        # performs calculate centroid and clustering until the system
        # becomes stable
        referrenceCluster = deepcopy(self.cluster)
        #for w in range (8):
        while (self.cluster == referrenceCluster):
            referrenceCluster = deepcopy(self.cluster)
            self.FillClusters()

            for i in range(self.K):
                self.CalcNewCentroid(i)

            #print len(self.cluster[0])
            #print len(self.cluster[1])
            #print len(self.cluster[2])
            #self.cluster = deepcopy(resetCluster)


        self.PlotClusters()


    def PlotClusters(self):
        np.set_printoptions(threshold=np.nan)
        # image to show clusters
        self.clusteredImage = deepcopy(self.origI)
        #print self.origI[50][100] # returns tupple
        randomColors = []
        # used for accuracy tests
        binaryImage = []
        binaryImage.append((0,0,0))
        binaryImage.append((255,255,255))
        # mask for clusters
        self.mask = deepcopy(self.I)
        # generate random colors for clusters
        for i in range(self.K):
            red = random.randrange(0,255)
            blue = random.randrange(0,255)
            green = random.randrange(0,255)
            randomColors.append((red,blue,green))

        # color assign each cluster
        for clustNum in range (self.K):
            for currentIndex in range(len(self.cluster[clustNum])):
                # gets coor. in tupple
                x = self.cluster[clustNum][currentIndex][0]
                y = self.cluster[clustNum][currentIndex][1]

                self.clusteredImage[y][x][0] = randomColors[clustNum][0]
                self.clusteredImage[y][x][1] = randomColors[clustNum][1]
                self.clusteredImage[y][x][2] = randomColors[clustNum][2]
                if (self.K == 2):
                    self.clusteredImage[y][x] = binaryImage[clustNum]
                # assigns px to corresponding cluster
                self.mask[y][x] = clustNum

        # creates binary image for algorithm analysis
        (_ ,self.binIm) = cv2.threshold(self.mask,0,255,cv2.THRESH_BINARY )
        # creates binary image of expected output image
        exOutput = cv2.imread("out1.jpg", 0)
        (_ ,self.testImageAEdge) = cv2.threshold(exOutput,0,255,cv2.THRESH_BINARY )
        # prints image in binary
        #print self.binIm


    def DisplayClusteredImage(self):
        cv2.imshow("clustered Image", self.clusteredImage)
        cv2.imshow("original Image", self.origI)
        cv2.waitKey(0)

    def CalcNewCentroid(self, clusterIndex):
        sumOfIntensities = 0
        # finds the average intensity and sets as new centroid
        for i in range(len(self.cluster[clusterIndex])):
            # px coor. that has intensity
            point = self.cluster[clusterIndex][i]
            x = point[0]
            y = point[1]

            sumOfIntensities += self.I[y][x]
        divZerofix = len(self.cluster[clusterIndex])
        #if (divZerofix == 0):
        #    avgIntensity = np.int(sumOfIntensities/np.inf)
        #else:
        avgIntensity = np.int(sumOfIntensities/len(self.cluster[clusterIndex]))

        # new centroid will be the coord with the intensity nearest to the
        # avg. intensity
        nwCent = self.cluster[clusterIndex][0]
        x = nwCent[0]
        y = nwCent[1]
        smallestDist = np.square(np.abs(avgIntensity - self.I[y][x]))
        for i in range(len(self.cluster[clusterIndex])):
            point = self.cluster[clusterIndex][i]
            x1 = point[0]
            y1 = point[1]

            if (np.square(np.abs(avgIntensity - self.I[y1][x1])) < smallestDist):
                nwCent = point
                smallestDist = np.square(np.abs(avgIntensity - self.I[y1][x1]))
        # replace previous centroid
        self.centroids[clusterIndex] = nwCent

    def AlgorithmAnalysis(self):
        totalPx = self.binIm.size
        h,w = self.binIm.shape

        ON_PIXEL = 255
        OFF_PIXEL = 0

        # calculates TP
        count = 0.0

        for i in range (0, h):
            for j in range (0, w):

                if (self.binIm[i][j] == ON_PIXEL and self.binIm[i][j] == self.testImageAEdge[i][j]):
                    count += 1

        TP = count / totalPx

        #calculates TN
        count = 0.0

        for i in range (0, h):
            for j in range (0, w):
                if (self.binIm[i][j] == OFF_PIXEL and self.binIm[i][j] == self.testImageAEdge[i][j]):
                    count += 1

        TN = count/totalPx

        # calculates FP
        count = 0.0

        for i in range (0, h):
            for j in range (0, w):
                if (self.binIm[i][j] == ON_PIXEL and self.binIm[i][j] != self.testImageAEdge[i][j]):
                    count += 1
        FP = count/totalPx

        # calculates FN f
        count = 0.0

        for i in range (0, h):
            for j in range (0, w):
                if (self.binIm[i][j] == OFF_PIXEL and self.binIm[i][j] != self.testImageAEdge[i][j]):
                    count += 1

        FN = count/totalPx
        
            # using pre calculated values, the edge comparison is performed
        SensitivityA = TP/(TP + FN)
    
        SpecificityA = TN/(TN + FP)
    
        PrecisionA = TP/(TP + FP)
    
        NegativePredictiveValueA = TN / (TN + FN)
    
        FallOutA = FP / (FP + TN)
    
        FNRA = FN / (FN + TP)
    
        FDRA = FP/(FP + TP)
    
        AccuracyA = (TP + TN)/(TP + FN + TN + FP)
    
        F_ScoreA = (2 * TP) / ( (2 * TP) + FP + FN)
    
        MCCA = ((TP * TN) - (FP * FN)) / \
               np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
        print ("\n\nSensitivity = %f\nSpecificity = %f\nPrecision = %f\nNegativePredictiveValue = %f\n"
               "FallOut = %f\nFNR = %f\nFDR = %f\nAccuracy = %f\nF_Score = %f\nMCC = %f\n" %
               (SensitivityA, SpecificityA, PrecisionA,
                NegativePredictiveValueA, FallOutA, FNRA,
                FDRA, AccuracyA, F_ScoreA, MCCA))
    
        return (SensitivityA, SpecificityA, PrecisionA, NegativePredictiveValueA, FallOutA, FNRA,
                FDRA, AccuracyA, F_ScoreA, MCCA)

