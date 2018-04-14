import cv2
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import itertools
#from os import walk
import os
import re
from PIL import Image

# tells np to print entire array
np.set_printoptions(threshold=np.nan)

class FaceRec :
    # Defines which feature vector to use
    USING_LBP_HISTOGRAM = 0
    USING_INTEGRAL_INTENSITIES = 1
    USING_IMAGE_INTENSITIES = 2
    USING_ALL = 3

    ''' Deprecated: Forced user to manually change directory on their computer
    # used to easily access all images
    # NOTE: for this to work on your computer you must change imgFolder location to the one
    # on your computer
    imgFolder = "C:\Users\Rich\OneDrive\Programming\Computer Vision\Assignment5\yalefaces"
    personFolder = ["\Person1", "\Person2", "\Person3", "\Person4", "\Person5", "\Person6", "\Person7"]
    imageName = ["\subject.jpg", "\subjectGlasses.jpg", "\subjectHappy.jpg", "\subjectLeftLight.jpg"
                 "\subjectNoGlasses.jpg", "\subjectNormal.jpg","\subjectRightLight.jpg", "\subjectSad.jpg",
                 "\subjectSleepy.jpg", "\subjectSurprised.jpg", "\subjectWink.jpg"]
    pic = cv2.imread(imgFolder + personFolder[0] + imageName[0], 0)

    usage:
        f = []
        for (dirpath, dirnames, filenames) in walk(self.imgFolder + self.personFolder[0]):
            f.extend(filenames)
            break

        print f
        print f[0]
    '''
    testingDir = "testingImages"
    trainingDir = "trainingImages"

    # gets lists of files in the directories
    traininglst = os.listdir(trainingDir)
    testinglst = os.listdir(testingDir)

    numNeighbors = 8 # because window is 3 * 3 for lbp, 8px around center
    regSize = 16 # size of window being taken by lbp matrix


    def __init__(self, choice):
        trainName = [] # names for the training images
        trainTag = []  # holds the tags (subject#)

        regularExpression = re.compile(r'\d+')
        count = 0
        # loop for each pic in training set
        for pic in self.traininglst:
            # tag and name maps by index
            trainTag.append([int(x) for x in regularExpression.findall(pic)][0])
            trainName.append(pic)
            # builds full image reference to be loaded
            picName = self.trainingDir + '/' + pic
            # use PIL to load image instead of cv2 because image ext. didn't load
            # properly using cv2.imread

            try:
                img = np.array(Image.open(picName).convert('L'))
            except:
                print("ERROR! " + pic + " File was not found! Trying next image\t")
                continue


            # uses choice passed in from user or main to perform analysis
            if (choice == self.USING_LBP_HISTOGRAM):
                featVec = self.LBP2(img)
                #print ("Performing LBP feature vector test")
            elif (choice == self.USING_INTEGRAL_INTENSITIES):
                featVec = self.IntegrateImage(img)
                #print ("Performing Integral Intensity feature vector test")
            elif (choice == self.USING_IMAGE_INTENSITIES):
                featVec = self.NormImage(img)
                #print ("Performing Image Intensity feature vector test")
            elif (choice == self.USING_ALL):
                intImVec = self.IntegrateImage(img)
                normImVec = self.NormImage(img)
                lbpVec = self.LBP2(img)
                totVec = np.zeros((1,lbpVec.shape[0] + intImVec.shape[0] + normImVec.shape[0]))
                totVec[:] = np.concatenate((lbpVec, intImVec, normImVec),axis=0)
                featVec = totVec
               # print ("Performing all feature vector test")

            if (count < 1):
                count = count + 1
                superFeatVector = np.zeros(featVec.shape)

            superFeatVector = np.vstack((superFeatVector,featVec))


        #cv2.imshow("The picture", self.pic)
        #stores all the names in directories

        # delete the first row because it is all zeros
        superFeatVector = np.delete(superFeatVector, 0, 0)

        # subtract feature vector from mean
        superFeatVector = superFeatVector - superFeatVector.mean()
        # form matrix by vector multiplication
        C = np.dot(superFeatVector, superFeatVector.transpose())
        # gets eigen values to be sent to pca
        eigenVal, eigenVect = np.linalg.eigh(C)

        V = self.PCA2( eigenVect, eigenVal, superFeatVector)

        # change to show eigen faces
        if (False):
            plt.gray()
            plt.figure(1)
            plt.imshow((superFeatVector.mean).reshape(img.shape))
            plt.figure(1)
            plt.imshow(V[0].reshape(img.shape))
            plt.show()

        # this is what is used to comapre differences
        trainer = np.dot(V, superFeatVector.T)

        testName = [] # names for the training images
        testTag = []  # holds the tags (subject#)


        count = 0
        # loop for each pic in training set
        for pic in self.testinglst:
            # tag and name maps by index
            testTag.append([int(k) for k in regularExpression.findall(pic)][0])
            testName.append(pic)
            # builds full image reference to be loaded
            picName = self.testingDir + '/' + pic
            # use PIL to load image instead of cv2 because image ext. didn't load
            # properly using cv2.imread
            try:
                img = np.array(Image.open(picName).convert('L'))
            except:
                print("ERROR! " + pic + " File was not found! Trying next image\t")
                continue


            # uses choice passed in from user or main to perform analysis
            if (choice == self.USING_LBP_HISTOGRAM):
                featVec = self.LBP2(img)
                #print ("Performing LBP feature vector test")
            elif (choice == self.USING_INTEGRAL_INTENSITIES):
                featVec = self.IntegrateImage(img)
                #print ("Performing Integral Intensity feature vector test")
            elif (choice == self.USING_IMAGE_INTENSITIES):
                featVec = self.NormImage(img)
                #print ("Performing Image Intensity feature vector test")
            elif (choice == self.USING_ALL):
                intImVec = self.IntegrateImage(img)
                normImVec = self.NormImage(img)
                lbpVec = self.LBP2(img)
                totVec = np.zeros((1,lbpVec.shape[0] + intImVec.shape[0] + normImVec.shape[0]))
                totVec[:] = np.concatenate((lbpVec, intImVec, normImVec),axis=0)
                featVec = totVec
                #print ("Performing all feature vector test")

            # if stack is nonexistant create stack
            if (count < 1):
                count = count + 1
                superFeatVector = np.zeros(featVec.shape)

            # creates stack of all images into one matrix with each row being a different training
            # image
            superFeatVector = np.vstack((superFeatVector,featVec))


        #cv2.imshow("The picture", self.pic)
        #stores all the names in directories

        # delete the first row because it is all zeros
        superFeatVector = np.delete(superFeatVector, 0, 0)

        # subtract feature vector from mean
        superFeatVector = superFeatVector - superFeatVector.mean()

        testor = np.dot(V, superFeatVector.T)

        # calc distance of trainer pic to each of the testing images
        lst = []

        for i in range (0, testor.shape[1]):
            index = 0
            deltaImage = []
            for j in range (0, trainer.shape[1]):
                A = testor[:,i]
                B = trainer[:,j]

                deltaImage.append(np.sum((testor[:,i] - trainer[:,j])**2))
            index = np.argmin(np.array(deltaImage))
            C = trainTag[index]
            lst.append(C)

        counter = 0

        # this is where we perform percentage calc.
        for i in range(len(lst)):
            if lst[i] == testTag[i]:
                count = count + 1

        print ("Count: %d" %(count))
        print ("testNum: %d" %(len(testTag)))

        # print results
        if (choice == self.USING_LBP_HISTOGRAM):
            print ("LBP feature vecture performed with an accuracy of %.01f"  %((np.float(count)/np.float((len(testTag))) * 100)) + " percent")
        elif (choice == self.USING_INTEGRAL_INTENSITIES):
            print ("Integral intensity feature vecture performed with an accuracy of %.01f"  %((np.float(count)/np.float((len(testTag))) * 100)) + " percent")
        elif (choice == self.USING_IMAGE_INTENSITIES):
            print ("Original image intensity feature vecture performed with an accuracy of %.01f"  %((np.float(count)/np.float((len(testTag))) * 100)) + " percent")
        elif (choice == self.USING_ALL):
            print ("Combination feature vecture performed with an accuracy of %.01f"  %((np.float(count)/np.float((len(testTag))) * 100)) + " percent")

        #print ("Count: %d" %(count))
        #print ("testNum: %d" %(len(testTag)))
        #print type((count/(len(testTag))))
        #print (np.float(count)/np.float((len(testTag))))
        #print ((count/len(testTag)) * 100)
        return
        # different feature vectors
        lbpFeatVec = self.LBP(self.pic)
        integralFeatVec = self.IntegrateImage(self.pic)
        normFeatVec = self.NormImage(self.pic)
        # concatenate lbpFeatVec integralFeatVec normFeatVec to form a super
        # feature vector
        superFeatVector = []
        superFeatVector.append(lbpFeatVec)
        superFeatVector.append(integralFeatVec)
        superFeatVector.append(normFeatVec)

        #concatenates all features into one array
        _superConcat = itertools.chain(lbpFeatVec, integralFeatVec, normFeatVec)
        superConcat = list(_superConcat)
        superConcat = np.asarray(superConcat)

        superFeatVector = np.asarray(superFeatVector).flatten()

        #projMat, varience, mean = pca(np.asarray(lbpFeatVec))

       # projMat, varience, mean = self.PCA(superFeatVector)
        projMat, varience, mean = self.PCA(superConcat)

    def ForAllImages(self):
        # performs image recognition and training for each subject
        for j in range (0, len(self.personFolder)):
            for i in range (0, len(self.imageName)):
                # I is loaded image
                try:
                    I = cv2.imread(self.imgFolder + self.personFolder[j] + self.imageName[i])
                except:
                    print("Image was not properly loaded. Trying next Image.")
                    continue
                # TODO: put case to catch exception where file was not properly loaded
                if (False):
                    continue
                # different feature vectors
                lbpFeatVec = self.LBP(I)
                integralFeatVec = self.IntegrateImage(I)
                normFeatVec = self.NormImage(I)
                # concatenate lbpFeatVec integralFeatVec normFeatVec to form a super
                # feature vector
                superFeatVector = []
                superFeatVector.append(lbpFeatVec)
                superFeatVector.append(integralFeatVec)
                superFeatVector.append(normFeatVec)

                superFeatVector = np.asarray(superFeatVector)
                projMat, varience, mean = pca(np.asarray(superFeatVector).flatten())
                #cv2.waitKey(0)
                print("made facerec")

    def NormImage(self, img):
        im = deepcopy(img)

        #normVect = np.zeros(img.shape)
        #normVect[:] = im[:, :]
        # flattens image
        normVect = np.reshape(img, img.shape[0]*img.shape[1])
        # calc. histogram for grayscale image and return
        #hist,bins = np.histogram(np.asarray(im).ravel(),256,[0,256])
        #normVect.append(hist)
        # return flattened image NOT histogram
        return normVect

    def LBP2(self, img):

        sPts = np.zeros((self.numNeighbors,2))
        for i in range(0,self.numNeighbors):

            sPts[i][0] = -1*np.sin((2*np.pi*i)/self.numNeighbors)
            sPts[i][1] = 1*np.cos((2*np.pi*i)/self.numNeighbors)


        height, width = img.shape

        miny=min(sPts[:,0])
        maxy=max(sPts[:,0])
        minx=min(sPts[:,1])
        maxx=max(sPts[:,1])

        # Block size, each LBP code is computed within a block of size bsizey*bsizex
        bsizey= np.ceil(max(maxy,0))-np.floor(min(miny,0))+1;
        bsizex= np.ceil(max(maxx,0))-np.floor(min(minx,0))+1;

        # coord in square
        oy= 1 - np.floor(min(miny,0))-1
        origx= 1 - np.floor(min(minx,0))-1

        # finds the difference between image size and window size
        xDiff = width - bsizex;
        yDiff = height - bsizey;

        # fill center px
        center = img[oy:oy+yDiff+1,origx:origx+xDiff+1]
        #d_C = double(C);

        # used to calculate neighborhood value
        bins = np.power(2,self.numNeighbors)

        resHolder = np.zeros((yDiff+1,xDiff+1)) # used to hold the result

        # find LBP matrix
        for i  in range(0,self.numNeighbors):
            y = sPts[i][0]+oy
            x = sPts[i][1]+oy

            floory = np.floor(y)
            ceily = np.ceil(y)
            roundy = round(y)
            floorx = np.floor(x)
            ceilx = np.ceil(x)
            roundx = round(x)
            # this is to check if it is needed to interpolate values
            if (abs(x - roundx) < 0.000021) and (abs(y - roundy) < 0.000021 ):
            # If interpolation is not needed
                interpSum = img[roundy:roundy+yDiff+1,roundx:roundx+xDiff+1]
                D = (interpSum >= center)*1
            else:
            # If interpolation is needed
                ty = y - floory;
                tx = x - floory;

                w1 = (1 - tx) * (1 - ty)
                w2 = (tx) * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty

                # calc interp val
                floorSum = w1*img[floory:floory+yDiff+1,floorx:floorx+xDiff+1]
                floorSum2 = w2*img[floory:floory+yDiff+1,ceilx:ceilx+xDiff+1]
                ceilSum = w3*img[ceily:ceily+yDiff+1,floorx:floorx+xDiff+1]
                CeilSum2 = w4*img[ceily:ceily+yDiff+1,ceilx:ceilx+xDiff+1]

                interpSum = floorSum + floorSum2 + ceilSum + CeilSum2

                #print C.shape

                D = (interpSum >= center) * 1

            # Update the result matrix.
            v = np.power(2,i)
            resHolder = resHolder + v * D
            # plt.gray()
            # plt.figure(1)


        # #compute the histogram of the LBP image across 8*8 block and concatenate all the histograms
        #windowsize_r = noofregions
        #windowsize_c = noofregions
        val = []
        for r in range(0,resHolder.shape[0]-self.regSize, self.regSize):
            for c in range(0,resHolder.shape[0]-self.regSize , self.regSize):
                win = resHolder[r:r+self.regSize,c:c+self.regSize]
                hist= np.histogram(win,2**self.numNeighbors,range =(0,2**self.numNeighbors),normed=True)[0]
                val = val + np.ndarray.tolist(hist)
        lbpFeatureVector = np.asarray(val)
        # change if you wish to see histograms

        if (False):
            plt.gray()
            plt.figure(4)
            plt.plot(val,color = 'g')
            plt.show()
        return lbpFeatureVector

    def IntegrateImage(self, img):
        im = deepcopy(img)

        intCalc = cv2.integral(im) # integral calculation by cv2
        integVector = np.zeros(img.shape) # allows values to be larger than 255

        # calc. histogram for integral image and return
        #hist,bins = np.histogram(np.asarray(integIm).ravel(),256,[0,256])

        #integVector = np.array(integIm).flatten() # doesnt work as needed
        integVector[:] = intCalc[1:,1:] # syntactical sugar for squishing matrix into vector
        integVector.reshape(img.size)
        #integVector.append(np.asarray(hist))

        #integralIm = deepcopy(img)

        height, width = np.array(img).shape

        ''' THIS MANUALLY PERFORMS INTEGRAL, IT TAKES TOO MUCH TIME
        # used to hold sum of intensities up until pixel
        sum = 0
        # loop to set area
        for j in range(0, height):
            for i in range (0, width):
                # loop to sum up area
                for b in range (0,j + 1):
                    for a in range (0, i + 1):
                        sum += img[b][a]
                sum = 0
                integralIm[j][i] = sum
        '''
        # return flattened integral image NOT histogram
        return integVector[0]

    def LBP(self, img):
        im = deepcopy(img)
        im = np.asarray(im) # used to calculate coded matrix
        lbpCodedMatrix = deepcopy(im) # used to calculate
        height, width = im.shape

        individualLBP = np.asarray([])
        temp = []

        powArray = [] # hold the powers of 2 to be multiplied
        count = -1
        # loop iterates over image (center pixel moving)
        for j in range (1, height - 1):
            for i in range (1, width - 1):
                # loops around center pixel
                for y in range (-1, 2):
                    for x in range (-1, 2):
                        # skip center pixel
                        if (x == 0 and y == 0):
                            continue
                        # if its less than thresh then its final value is zero
                        if (im[j][i] >  im[j + y][i + x]):
                            powArray.append(0)
                        else:
                            powArray.append(1)
                    count += 1
                # multiplies the thresholded window by pwoers of 2
                for q in range (0, len(powArray)):
                    powArray[q] = powArray[q] << q
                # forms LBP coded matrix w/ center calculated as sum of  neighbors
                index = 0
                for y in range (-1, 2):
                    for x in range (-1, 2):
                        if (x == 0 and y == 0):
                             lbpCodedMatrix[j + y][i + x] = sum(powArray)
                        else:
                            lbpCodedMatrix[j + y][i + x] = powArray[index]
                # resets powerArray for next iteration
                powArray = []

        # creates histogram using window size (can vary as a square power of 2, 2** x 2**)
        windowSize = 8

        pixelContainer = [] # holds pixels in window
        # holds windows in image, will be used to calculate histograms of each window
        # and then concatenate them together
        windowContainer = []
        # moves center pixel, grabs window and moves center pixel
        # no overlapping
        for j in range (0 + windowSize / 2, height - (windowSize / 2), windowSize):
            for i in range (0 + windowSize / 2, height - (windowSize / 2), windowSize):
                # windowSize out of bounds, maybe grab remainder windows?
                if (j + (windowSize / 2) >= height or i + (windowSize / 2) >= width):
                    break

                # grab window of pixels
                for y in range (-(windowSize / 2), (windowSize/2)):
                    for x in range (-(windowSize / 2), (windowSize/2)):
                        pixelContainer.append(lbpCodedMatrix[j + y][i + x])
                # add window to window container and clear pixels
                windowContainer.append(pixelContainer)
                pixelContainer = []

                cv2.imshow("LBP coded matrix", lbpCodedMatrix)
        # formatted better
        #hist,bins = np.histogram(img.ravel(),256,[0,256])

        # concatenation of all lbp widow histograms
        lbpFeatureVector = []
        # creates histogram for window
        for k in range (0, len(windowContainer)):
            hist,bins = np.histogram(np.asarray(windowContainer[k]).ravel(),256,[0,256])
            lbpFeatureVector.append(np.asarray(hist))

        #print (hist)
        #print len(hist)
        #print sum(hist)
        #print width * height
        # makes and displays histogram
        plt.hist(np.asarray(windowContainer[50]).ravel(),256,[0,256]); plt.show()
        return lbpFeatureVector


    def PCA2(self, eigVec, eigVal, superFeatVec):
        face = np.dot(superFeatVec.T, eigVec).T
        V = face[::-1] # gets highest values

        try:
            S = np.sqrt(eigVal)[::-1] # set descending order
        except:
            print("Something happened in PCA2 calculating S")

        for i in range (0,V.shape[1]):
            V[:,i] /= S

        # returns varience
        return V[0:20,:]

    def PCA(self, X):
        # only works with flattened data
        # X contains training data
        #X = np.asarray(X).flatten()

        # get dimensions
        num_data, dim = X.shape

        # center data
        mean_X = X.mean(axis=0)
        X = X * mean_X

        if dim > num_data:
            M = np.dot(X,X.T)
            e, EV = np.linalg.eigh(M)
            tmp = np.dot(X.T,EV).T
            V = tmp[::-1]
            S = np.sqrt(e)[::-1]
        else:
            M = np.dot(X,X.T)
            e, EV = np.linalg.eigh(M)
            tmp = np.dot(X.T,EV).T
            V = tmp[::-1]
            S = np.sqrt(e)[::-1]


        for i in range (V.shape[1]):
            V[:,i] /= S
        else:
            U,S,V = np.linalg.svd(X)
            V = V[:num_data]

        return  V,S,mean_X


class pca:
    def __init__(self, X):
        """ Principal Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance and mean.
        """
        # get dimensions
        num_data,dim = X.shape
        # center data
        mean_X = X.mean(axis=0)
        X = X - mean_X
        if dim>num_data:
            # PCA - compact trick used
            M = np.dot(X,X.T) # covariance matrix
            e,EV = np.linalg.eigh(M) # eigenvalues and eigenvectors
            tmp = np.dot(X.T,EV).T # this is the compact trick
            V = tmp[::-1] # reverse since last eigenvectors are the ones we want
            S = np.sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
            for i in range(V.shape[1]):
                V[:,i] /= S
        else:
            # PCA - SVD used
            U,S,V = np.linalg.svd(X)
            V = V[:num_data] # only makes sense to return the first num_data

            # return the projection matrix, the variance and the mean
            return V, S, mean_X