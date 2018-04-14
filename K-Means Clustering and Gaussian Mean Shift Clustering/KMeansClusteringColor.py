import cv2
import numpy as np
from copy import copy, deepcopy
from matplotlib import pyplot as plt
import random


class KmeansColor:
    def __init__(self, img, K):
        self.origI = img
        # K = # of clusters
        self.K = K
        # Igrey = image in grayscale
        self.Igrey = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        self.I = deepcopy(img)
        # centroids = the mean of each cluster, there are k centroids
        self.centroids  = []
        # create cluster array
        self.cluster = []
        # note the index of each cluster array matches the number
        # of clusters
        self.PerformAlg()
        #show results
        self.DisplayClusteredImage()


    def PickRandomCentroids(self):
        # the algorithm must be initialized with k random intensities
        # from the dataset
        self.h,self.w = self.Igrey.shape
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
                minDistance = self.EuclideanDistance(pointIntensity,centroidStartIntensity)#np.abs(pointIntensity - centroidStartIntensity)
                #print minDistance
                # compare with distance to all centroids
                for centroidIndex in range(len(self.centroids)):
                    compareValue = self.EuclideanDistance(pointIntensity,self.I[self.centroids[centroidIndex][1]][self.centroids[centroidIndex][0]])#np.abs(pointIntensity - self.I[self.centroids[centroidIndex][1]][self.centroids[centroidIndex][0]])
                    if ( compareValue < minDistance):
                        minDistance = compareValue
                        clusterToBeAssignedTo = centroidIndex
                # final cluster assignment
                self.cluster[clusterToBeAssignedTo].append((i,j))

    def EuclideanDistance(self, point1, point2):

        colorDistance = np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]) + np.square(point1[2] - point2[2]))
        return np.round(colorDistance)

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
            #print ("sum = %d" %((len(self.cluster[0]))+len(self.cluster[1])+ len(self.cluster[2])))

            #print len(self.cluster[0])
            #print len(self.cluster[1])
            #print len(self.cluster[2])
            #self.cluster = deepcopy(resetCluster)

            #print("\n\n")

        self.PlotClusters()


    def PlotClusters(self):
        np.set_printoptions(threshold=np.nan)
        # image to show clusters
        self.clusteredImage = deepcopy(self.origI)
        #print self.origI[50][100] # returns tupple
        randomColors = []
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

                self.mask[y][x] = clustNum
        #print self.mask



        #print self.origI

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

            #sumOfIntensities += self.I[y][x]
            a = self.I[y][x][0]
            b = self.I[y][x][1]
            c = self.I[y][x][2]
            sumOfIntensities += np.sqrt(a**2 + b**2 + c**2)
        divZerofix = len(self.cluster[clusterIndex])
        #if (divZerofix == 0):
        #    avgIntensity = np.int(sumOfIntensities/np.inf)
        #else:
        #print ("sum of intensities: %1f\nlenOfCluster: %d" %(sumOfIntensities,len(self.cluster[clusterIndex])))

        avgIntensity = np.int(sumOfIntensities/len(self.cluster[clusterIndex]))

        # new centroid will be the coord with the intensity nearest to the
        # avg. intensity
        nwCent = self.cluster[clusterIndex][0]
        x = nwCent[0]
        y = nwCent[1]

        colorMagnitude = np.sqrt(np.square(self.I[y][x][0]) + np.square(self.I[y][x][1]) + np.square(self.I[y][x][2]))
        #smallestDist = np.square(np.abs(avgIntensity - self.I[y][x]))
        smallestDist = np.square(np.abs(avgIntensity - colorMagnitude))

        for i in range(len(self.cluster[clusterIndex])):
            point = self.cluster[clusterIndex][i]
            x1 = point[0]
            y1 = point[1]
            #print avgIntensity
            #print smallestDist
            colorMagnitude = np.sqrt(np.square(self.I[y][x][0]) + np.square(self.I[y][x][1]) + np.square(self.I[y][x][2]))
            if (np.square(np.abs(avgIntensity - colorMagnitude)) < smallestDist):
                nwCent = point
                smallestDist = np.square(np.abs(avgIntensity - self.I[y1][x1]))
        # replace previous centroid
        self.centroids[clusterIndex] = nwCent


