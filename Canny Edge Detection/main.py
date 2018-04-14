from PIL import Image
import numpy as np
from scipy.misc import imread
import scipy.ndimage

import cv2
from copy import copy, deepcopy
import sys
from math import pi, sqrt, exp
import math
import matplotlib.pyplot as plt

##########################################################################
## Function Name: Q1
## Function Desc.: Operations needed for question 1
## Function Return: none
##########################################################################
def Q1():
    # changes the  print settings to show all values in array
    np.set_printoptions(threshold='nan')
    # reads i
    I = Image.open("input.jpg")
    I2 = cv2.imread("input2.jpg", 0)
    I3 = cv2.imread("input_image.jpg", 0)
    # image 1 with different sigma
    im_25 = CannyEdgeDetection(I, .25,3)
    im_50 = CannyEdgeDetection(I, .50,3)
    im_75 = CannyEdgeDetection(I, .75,3)
    # image 2 with different sigma
    im2_25 = CannyEdgeDetection(I2, .25,3)
    im2_50 = CannyEdgeDetection(I2, .50,3)
    im2_75 = CannyEdgeDetection(I2, .75,3)
    # image 3 with different sigma
    im3_25 = CannyEdgeDetection(I3, .25,3)
    im3_50 = CannyEdgeDetection(I3, .50,3)
    im3_75 = CannyEdgeDetection(I3, .75,3)

    # Shows all 3 images with their respective sigmas
    # NOTE: the best sigma overall was .75
    cv2.imshow("Image1: 25", im_25)
    cv2.imshow("Image1: 50", im_50)
    cv2.imshow("Image1: 75", im_75)

    cv2.imshow("Image2: 25", im2_25)
    cv2.imshow("Image2: 50", im2_50)
    cv2.imshow("Image2: 75", im2_75)

    cv2.imshow("Image3: 25", im3_25)
    cv2.imshow("Image3: 50", im3_50)
    cv2.imshow("Image3: 75", im3_75)

    cv2.waitKey(0)
    return

##########################################################################
## Function Name: CannyEdgeDetection
## Function Desc.: Performs my implementation of Canny Edge Detection
## Function Arguments: I = cv2 image, sigma = gaus arg, maskLength = len
## Function Return: image as np.array
##########################################################################
def CannyEdgeDetection(I, sigma, maskLength):
    # used to get image dimensions
    height, width = np.array(I).shape

    # creates 1d gaussian of the passed in Length
    G = ProfGoss(maskLength,sigma)
    pic = deepcopy(I)

    # creates 1D Gaussian masks for the derivitive of the function in the x and y directions respectively
    # Note* the same derivative function will be used for dx && dy because they are essentially the same
    # but treated different in orientation only
    Gx = gausXDeriv(maskLength, sigma)
    Gy = gausYDeriv(maskLength, sigma)

    #Gx = [-1,0,1]
    #Gy = [-1,0,1]
    # Gets image after being convolved with mask in x-dir
    Ix = ProfConvolveX(I, G, width, height)
    # Gets image after being convolved with mask in y-dir
    Iy = ProfConvolveY(I, G, width, height)

    # convolves derivatives into already convolved image
    # converts to np.array for later calculations
    IPrimeX = np.array(ProfConvolveX(Ix,Gx,width,height))
    IPrimeY = np.array(ProfConvolveY(Iy,Gy,width,height))

    # magnitude of edge response calculation
    tempArray = IPrimeX * IPrimeX + IPrimeY *IPrimeY
    M = np.sqrt(tempArray)
    cv2.imshow("Magnitude", M)
    # determines the gradient for suppresion and converts to degrees
    direction = np.arctan2(IPrimeY, IPrimeX) * 180 / np.pi

    # stores result of suppression to be used with thresholding
    nonMaxResult = NonMaximumSuppression(M, direction)
    cv2.imshow("Final edge trace", nonMaxResult)
    # pauses cv2 to allow image to be seen
    cv2.waitKey(0)
    #cv2.imshow("binary", cv2.threshold(nonMaxResult,127,255,cv2.THRESH_BINARY))
    return nonMaxResult

##########################################################################
## Function Name: ProfGoss
## Function Desc.: Returns a 1D gaussian mask of len size and varied sigma
## Function Arguments: I = PIL image, sigma = gaus arg, maskLength = len
##########################################################################
def ProfGoss(size, sigma):
    # forces size to be odd
    if (size % 2 == 0):
        size -= 1
    # array to hold gaussian elements
    mask = []
    # sigma ^ 2
    sigmaSquare = sigma * sigma
    # gaussian formula
    mult = 1.0/math.sqrt(2.0 * math.pi * sigmaSquare)
    # loop to populate mask
    for i in range (-size/2+1, size/2+1):
        mask.append(mult * exp(-i * i/(2 * sigmaSquare)))
    return mask

##########################################################################
## Function Name: ProfConvolveX
## Function Desc.: Convolves image in X direction and returns result
## Function Arguments: image = Pil, mask = 1d mask, w = width, h = height
##########################################################################
def ProfConvolveX(image, mask, w, h):
    # creates deep copy of image
    #newImage = image.copy()
    #newImagePixels = newImage.load()
    #pixels = image.load()

    conversion = deepcopy(np.array(image))
    pixels = deepcopy(np.array(image))

    # loop for width then length
    for i in range (0, h):
        for j in range (0, w):
            sum = 0.0
            count = 0
            lastCenter = 0
            # used to calculate offset for array
            for k in range (-len(mask)/2 + 1, len(mask)/2 + 1):
                # actual index in image arr
                nj = j + k
                # in the event offset is out of bounds
                if (nj < 0 or nj >= w):
                    continue

                lastCenter = nj

                # performs array index * mask index and adds to sum
                # k + len(mask)/2 = center + offset -k -> k
                sum += mask[k + len(mask)/2] * pixels[i,nj]

            conversion[i,lastCenter] = sum / len (mask)

            sum = 0
    return conversion

##########################################################################
## Function Name: ProfConvolveY
## Function Desc.: Convolves image in Y direction and returns result
## Function Arguments: image = Pil, mask = 1d mask, w = width, h = height
##########################################################################
def ProfConvolveY(image, mask, w, h):
    # creates deep copy of image
    conversion = deepcopy(np.array(image))
    pixels = deepcopy(np.array(image))

    #formatted pixel[width,height]
    for j in range (0, w):
        for i in range (0, h):
            sum = 0
            count = 0
            lastCenter = 0
            for k in range (-len(mask)/2 + 1, len(mask)/2 + 1):
                nj = i + k
                if (nj < 0 or nj >= h):
                    continue
                lastCenter = nj
                #count += mask[k + len(mask)/2]
                sum += mask[k + len(mask)/2] * pixels[nj,j]

            conversion[lastCenter,j] = sum / len(mask)

    return conversion

##########################################################################
## Function Name: NonMaximumSuppression
## Function Desc.: Performs nonMaximumSuppression
## image and mask
##########################################################################
def NonMaximumSuppression(magnitude, dir):
    dirCpy = deepcopy(dir)
    magCpy = deepcopy(magnitude)

    # obtains img dimensions
    size = magnitude.shape # formatted (Height, Width)
    # sets width to image width
    width = size[1]
    # sets height to image height
    height = size[0]

    edge = deepcopy(magCpy)
    # rounds angles to nearest degree based on 22.5(degree) difference
    for a in range (0, height):
        for b in range (0, width):
            #sets edges to blank white canvas
            #edge[a][b] = 255

            q = dirCpy[a][b]
            if (q <= 180 and q > 124):
                dirCpy[a][b] = 135
            if (q <= 124 and q > 101):
                dirCpy[a][b] = 112.5
            if (q <= 101 and q > 79):
                dirCpy[a][b] = 90
            if (q <= 79 and q > 56):
                dirCpy[a][b] = 67.5
            if (q <= 56 and q > 34):
                dirCpy[a][b] = 45
            if (q <= 34 and q > 11):
                dirCpy[a][b] =22.5
            if (q <= 11 and q > -350):
                dirCpy[a][b] = 0
    #print dirCpy
    # since all angles have been rounded it is easy to switch between possibilities
    # check pixel along gradient direction to determine if it is local maximum
    # NOTE: the edge is perpendicular to gradient
    for y in range (0, height):
        for x in range (0, width):
            #TODO: 112.5
            if (dirCpy[y][x] == 135):
                if ((y + 1 < height and x + 1 < width) and (y - 1 >= 0 and x - 1 >= 0)):
                    if (magCpy[y][x] > magCpy[y + 1][x - 1] and magCpy[y][x] > magCpy[y - 1][x + 1]):
                        magCpy[y][x] = 0
            if (dirCpy[y][x] == 112.5):
                if (y + 1 < height and y - 1 >= 0):
                    if (magCpy[y][x] > magCpy[y + 1][x] and magCpy[y][x] > magCpy[y - 1][x]):
                        magCpy[y][x] = 0
            if (dirCpy[y][x] == 90):
                if (y + 1 < height and y - 1 >= 0):
                    if (magCpy[y][x] > magCpy[y + 1][x] and magCpy[y][x] > magCpy[y - 1][x]):
                        magCpy[y][x] = 0
            if (dirCpy[y][x] == 67.5):
                if ((y + 2 < height and x + 1 < width) and (y - 2 >= 0 and x - 1 >= 0)):
                    if (magCpy[y][x] > magCpy[y + 2][x + 1] and magCpy[y][x] > magCpy[y - 2][x + 1]):
                        magCpy[y][x] = 0
            if (dirCpy[y][x] == 45):
                if ((y + 1 < height and x + 1 < width) and (y - 1 >= 0 and x - 1 >= 0)):
                    if (magCpy[y][x] > magCpy[y + 1][x + 1] and magCpy[y][x] > magCpy[y - 1][x - 1]):
                        magCpy[y][x] = 0
            if (dirCpy[y][x] == 22.5):
                if ((y + 1 < height and x + 2 < width) and ( y - 1 >= 0 and x - 2 >= 0)):
                    if (magCpy[y][x] > magCpy[y + 1][x + 2] and magCpy[y][x] > magCpy[y - 1][x - 2]):
                        magCpy[y][x] = 0
            if (dirCpy[y][x] == 0):
                if (x + 1 < width and x - 1 >= 0):
                    if (magCpy[y][x] > magCpy[y][x + 1] and magCpy[y][x] > magCpy[y][x - 1]):
                        magCpy[y][x] = 0

    highThresh = 13
    lowThresh = 1
    cv2.imshow("before edge detection",magCpy)

    edgeDetect(magCpy, highThresh,lowThresh)
    cv2.imshow("after edge detection", magCpy)


            #print magCpy
    #remove points below low threshold
    for y in range (0, height):
        for x in range (0, width):
            if (magCpy[y][x] < lowThresh):
                magCpy[y][x] = 0

    cv2.imshow("After minimum thresholding after edge detection",magCpy)

    retArray = deepcopy(magCpy)
    #remove points not near high threshold
    for y in range (0, height):
        for x in range (0, width):
            if (y + 1 < height and x + 1 < width):
                if (magCpy[y+1][x+1] >= highThresh):
                    continue
            if (x + 1 < width):
                if (magCpy[y][x+1] >= highThresh):
                    continue
            if (y - 1 >= 0 and x + 1 < width):
                if (magCpy[y-1][x+1] >= highThresh):
                    continue
            if (y - 1 >= 0):
                if (magCpy[y-1][x] >= highThresh):
                    continue
            if (y - 1 >= 0 and x - 1 >= 0):
                if (magCpy[y-1][x-1] >= highThresh):
                    continue
            if (x - 1 >= 0):
                if (magCpy[y][x-1] >= highThresh):
                    continue
            if (x - 1 >= 0 and y + 1 < height):
                if (magCpy[y+1][x-1] >= highThresh):
                    continue
            if (y + 1 < height):
                if (magCpy[y+1][x] >= highThresh):
                    continue
            magCpy[y][x] = 0

    #cv2.imshow("edges",edge)
    return retArray

##########################################################################
## Function Name: edgeDetect
## Function Desc.: detects edges
## Function Arguments: mat = matrix, tUpper = higher threshold, tLower = lower threshold
##########################################################################
def edgeDetect(mat, tUpper, tLower):
    matrix = np.array(mat)
    rows, cols = matrix.shape

    edges = deepcopy(matrix)

    # make edges all black
    for x in range (0, cols):
        for y in range(0, rows):
            edges[y][x] = 0

    for x in range (0, cols):
        for y in range(0, rows):
            if (matrix[y][x] >= tUpper):
                followEdges(x,y,matrix,tUpper,tLower, edges)


##########################################################################
## Function Name: followEdges
## Function Desc.: follows edges in an attempt to connect pixels
## Function Arguments: x,y = index, matrix = image, tUpper,tLower = thresholds
##                           edges = marked pixels
##########################################################################
def followEdges(x,y,matrix,tUpper,tLower, edges):
    #print("x: %d y: %d" %(x,y))
    sys.setrecursionlimit(100)
    #set point to white
    matrix[y][x] = 255
    edges[y][x] = 255
    height, width = matrix.shape

    #base case
    if (x >= width or y >= height):
        return

    deepestRecursion = 33
    if (x >= deepestRecursion or y >= deepestRecursion):
        sys.setrecursionlimit(1000)
        return
    for i in range (-1, 2):
        for j in range (-1, 2):
            if (i == 0 and j == 0):
                continue
            if ((x + i >= 0) and (y + j >= 0) and (x + i <= width) and (y + j <= height)):
                if((edges[y + j][j + i]) > tLower and edges[y + j][j + i] != 255):
                    followEdges(x + i, y + j, matrix, tUpper, tLower, edges)



##########################################################################
## Function Name: Zero Padding
## Function Desc.: adds zeros to array sides
## Function Arguments: None
##########################################################################
def ZeroPadding(arr):
    print ""

##########################################################################
## Function Name: Zero Padding
## Function Desc.: adds zeros to array sides
## Function Arguments: None
##########################################################################
def ResidualImages():
    print "Creating Residual Images"

##########################################################################
## Function Name: Entropy
## Function Desc.: Performs Entropy process in its entirety
## Function Arguments: None
##########################################################################
def Entropy():
    # the names of the individual pictures
    image1 = "input.jpg"
    image2 = "input2.jpg"
    image3 = "input4.jpg"

    # performs entropy operations and gains thresholds for each image
    image1Threshold = EntropyOp(image1)
    image2Threshold = EntropyOp(image2)
    image3Threshold = EntropyOp(image3)
    # finalizes the process by thresholding the image
    BinifyImage(image1, image1Threshold)
    BinifyImage(image2, image2Threshold)
    BinifyImage(image3, image3Threshold)

##########################################################################
## Function Name: BinifyImage
## Function Desc.: converts image to binary 0, 255 on and off
## Function Arguments: imgName = (string) img name, thres = threshold amnt (int)
##########################################################################
def BinifyImage(imgName, thres):
    # opens image
    img = cv2.imread(imgName, 0)
    # ges image dimensions
    height,width = img.shape

    imcpy = deepcopy(img)
    # offset that gives images best look
    # threshOffset = 60
    threshOffset = 0
    # loop turns pixels on and off based on value
    for i in range (0, height):
        for j in range (0, width):
            if (imcpy[i][j] >= thres + threshOffset):
                imcpy[i][j] = 255
            else:
                imcpy[i][j] = 0

    cv2.imshow("Binary manually", imcpy)
    # uses cv2 along with calculated threshold
    (thresh, im_bw) = cv2.threshold(img,thres,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("Binary with cv2", im_bw)
    np.set_printoptions(threshold='nan')
    print im_bw
    cv2.waitKey(0)
##########################################################################
## Function Name: EntropyOp
## Function Desc.: Quantitative Evaluation of Edge Detector
## Function Arguments: None
##########################################################################
def EntropyOp(imageName):

    # imports image and forces it to flatten and become grayscale
    #imageTest = Image.open(imageName).convert('L')

    # imports image and forces it to flatten and become grayscale
    img = cv2.imread(imageName, 0)
    # calculates histograram
    #hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #hist = cv2.calcHist([img],[0],None,[256],[0,256])
    # formatted better
    hist,bins = np.histogram(img.ravel(),256,[0,256])
    # makes and displays histogram
    plt.hist(img.ravel(),256,[0,256]); plt.show()
    image = imread(imageName)
    # dimenstions of image
    height,width = image.shape
    # area or number of pixels in image
    numPx = width * height
    # array to hold entropy
    valArray = []


    # CORRECTION: incorrect way of displaying histogram
    #n, bins, patches = plt.hist( histArray, bins=255, range=(0,255), histtype='step')
    #plt.xlabel("Value")
    #plt.ylabel("Frequency")
    #plt.show()

    # copies histogram as a float
    # probArray = probability of each pixel
    probArray = deepcopy(hist.astype(float))
    # normalizes the array, sum of all elements = 1
    for i in range (0,len(hist)):
        probArray[i] /= numPx

    # number to be added to zero to prevent divide by zeros
    theta = 1e-7

    # copies prob array and sets all values to 0 (float)
    P_T = deepcopy(probArray)
    for w in range (0, len(probArray)):
        P_T[w] = 0.0

    # calculate pt and store in array
    for i in range (0, len(probArray)):
        for j in range (0, i + 1):
            P_T[i] += probArray[j]


    # Store classes A and B of probability
    A = []
    B = []

    # calculate A class, theta prevent divide 0 error
    for i in range(0, len(probArray)):
        A.append(probArray[i]/(P_T[i] + theta))
        #print ("%f/(%f + %f) = %f" % (probArray[i], P_T[i], theta, probArray[i]/(P_T[i] + theta)))

    # calculate B class, theta prevent divide 0 error
    for i in range(0, len(probArray)):
        # prevents out of bounds error
        if (i + 1 >= len(probArray) - 1):
            B.append(theta)
            continue
        B.append(probArray[i+1]/(1.0 - P_T[i] + theta))
        #print ("B = %f/(%f + %f) = %f" % (probArray[i], P_T[i], theta, probArray[i]/(P_T[i] + theta)))


    # used to hold HA and HB
    HA = 0.0
    HB = 0.0

    # H(A)
    for i in A:
        it = -1 * i * np.log2(i + theta)
        HA += it
        #print ("HA: %f" %HA)

    # H(B)
    for i in B:
        it = -1 * i * np.log2(i + theta)
        HB += it
        #print ("HB: %f" %HB)

    # total entropy int not int[]
    HT = HA + HB

    print ("H(T) = %d" %HT)
    return HT


##########################################################################
## Function Name: Q2
## Function Desc.: Quantitative Evaluation of Edge Detector
## Function Arguments: None
##########################################################################
def Q2():

    name1 = "input_image.jpg"
    #name1 = "input.jpg"
    name2 = "input_image2.jpg"
    name3 = "input_image3.jpg"
    # reads in test images and converts to grayscale
    testImageA = Image.open(name1).convert('L')
    testImageB = Image.open(name2).convert('L')
    testImageC = Image.open(name3).convert('L')
    #testImageC = Image.open("input_image.jpg").convert('L')
    # reads in edge map for correlating images as grayscale
    testImageAEdge = np.array(Image.open("output_image.png").convert('L'))
    testImageBEdge = np.array(Image.open("output_image2.png").convert('L'))
    testImageCEdge = np.array(Image.open("output_image3.png").convert('L'))

    # opens same images for easy dimension access in form array[y][x]
    # total is area of rectangle or image px amnt.
    testImageAInfo = imread("input_image.jpg").shape
    heightA = testImageAInfo[0]
    widthA = testImageAInfo[1]
    totalA = widthA * heightA
    testImageBInfo = imread("input_image2.jpg").shape
    heightB = testImageBInfo[0]
    widthB = testImageBInfo[1]
    totalB = widthB * heightB
    testImageCInfo = imread("input_image3.jpg").shape
    heightC = testImageCInfo[0]
    widthC = testImageCInfo[1]
    totalC = widthC * heightC

    '''
    # _ prevents it from being a tuple, converts image to binary
    _ ,x = cv2.threshold(nonMaxResult,7,1,cv2.THRESH_BINARY)
    print x
    '''
    # to be used with gaussian and derivative calculations best sigma is .501
    sigma = .501
    # length of masks
    maskLength = 3

    A = CannyEdgeDetection(testImageA, sigma, maskLength)
    B = CannyEdgeDetection(testImageB, sigma, maskLength)
    C = CannyEdgeDetection(testImageC, sigma, maskLength)

    #cv2.imshow("A", A)
    cv2.waitKey(0)

    if (True):
        # converts both edge map and image to binary (redundant because cannyEdgeDetection does this)
        (_ ,binA) = cv2.threshold(A,0,255,cv2.THRESH_BINARY )
        (_ , testABin) = cv2.threshold(testImageAEdge,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        print ("Image 1 rankings")
        QEED(binA,testABin,widthA,heightA)

        (_ ,binB) = cv2.threshold(B,0,255,cv2.THRESH_BINARY)
        (_ , testBBin) = cv2.threshold(testImageBEdge,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        print ("Image 2 rankings")
        QEED(binB,testBBin,widthB,heightB)

        (_ ,binC) = cv2.threshold(C,0,255,cv2.THRESH_BINARY)
        (_ , testCBin) = cv2.threshold(testImageCEdge,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        print ("Image 3 rankings")
        QEED(binC,testCBin,widthC,heightC)


        cv2.waitKey(0)

    print ("Image 1 rankings with salt and pepper")
    # add noise to picture, salt and pepper then gaussian
    noisyTestA = Noice("gauss", Noice("s&p",np.array(testImageA)))
    noisyA = CannyEdgeDetection(noisyTestA, sigma, maskLength)
    # changes the  print settings to show all values in array
    np.set_printoptions(threshold='nan')
    # loop forces image to binary because cv2 complained about noised image
    for i in range (0, heightA):
        for j in range (0, widthA):
            if (noisyA[i][j] != 0):
                noisyA[i][j] = 255

    QEED(noisyA,testImageAEdge, widthA, heightA)


##########################################################################
## Function Name: QEED
## Function Desc.: Quantitative Evaluation of Edge Detector operations
## Function Arguments: binA = binary of original image, testImageAEdge =
##                          edge map to be compared, (w, h) = width height
##########################################################################
def QEED (binA,testImageAEdge, w, h):
    # area for rect or num px.
    totalA = float(w * h)
    ON_PIXEL = 255
    OFF_PIXEL = 0

    # shape = [Width, height]
    # calculates TP for A
    countA = 0.0
    for i in range (0, h):
        for j in range (0, w):
            if (binA[i][j] == ON_PIXEL and binA[i][j] == testImageAEdge[i][j]):
                countA += 1

    TP_A = countA/ totalA

    #calculates TN for A
    countA = 0.0

    for i in range (0, h):
        for j in range (0, w):
            if (binA[i][j] == OFF_PIXEL and binA[i][j] == testImageAEdge[i][j]):
                countA += 1

    TN_A = countA/totalA

    # calculates FP for A
    countA = 0.0

    for i in range (0, h):
        for j in range (0, w):
            if (binA[i][j] == ON_PIXEL and binA[i][j] != testImageAEdge[i][j]):
                countA += 1
    FP_A = countA/totalA

    # calculates FN for A, B, C
    countA = 0.0

    for i in range (0, h):
        for j in range (0, w):
            if (binA[i][j] == OFF_PIXEL and binA[i][j] != testImageAEdge[i][j]):
                countA += 1

    FN_A = countA/totalA

    print ("TP_A = %f\nTN_A = %f\nFP_A = %f\nFN_A = %f" %(TP_A, TN_A, FP_A, FN_A))

    # using pre calculated values, the edge comparison is performed
    SensitivityA = TP_A/(TP_A + FN_A)

    SpecificityA = TN_A/(TN_A + FP_A)

    PrecisionA = TP_A/(TP_A + FP_A)

    NegativePredictiveValueA = TN_A / (TN_A + FN_A)

    FallOutA = FP_A / (FP_A + TN_A)

    FNRA = FN_A / (FN_A + TP_A)

    FDRA = FP_A/(FP_A + TP_A)

    AccuracyA = (TP_A + TN_A)/(TP_A + FN_A + TN_A + FP_A)

    F_ScoreA = (2 * TP_A) / ( (2 * TP_A) + FP_A + FN_A)

    MCCA = ((TP_A * TN_A) - (FP_A * FN_A)) / \
           math.sqrt((TP_A + FP_A) * (TP_A + FN_A) * (TN_A + FP_A) * (TN_A + FN_A))

    print ("\n\nSensitivity = %f\nSpecificity = %f\nPrecision = %f\nNegativePredictiveValue = %f\n"
           "FallOut = %f\nFNR = %f\nFDR = %f\nAccuracy = %f\nF_Score = %f\nMCC = %f\n" %
           (SensitivityA, SpecificityA, PrecisionA,
            NegativePredictiveValueA, FallOutA, FNRA,
            FDRA, AccuracyA, F_ScoreA, MCCA))

    return (SensitivityA, SpecificityA, PrecisionA, NegativePredictiveValueA, FallOutA, FNRA,
            FDRA, AccuracyA, F_ScoreA, MCCA)


##########################################################################
## Function Name: gauss
## Function Desc.: returns the 1 dimensional gausian mask
##########################################################################
def gauss(n=43,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

##########################################################################
## Function Name: gausXDeriv
## Function Desc.: returns the 1 dimensional gausian mask
##########################################################################
def gausXDeriv(n=43,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return [-x / ((sigma ** 3) * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

##########################################################################
## Function Name: gausYDeriv
## Function Desc.: returns the 1 dimensional gausian mask
##########################################################################
def gausYDeriv(n=43,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return [-y / ((sigma ** 3) * sqrt(2*pi)) * exp(-float(y)**2/(2*sigma**2)) for y in r]

##########################################################################
## Function Name: Noice
## Function Desc.: generates noise in numpy images and returns disturbed
##                  image
##########################################################################
def Noice(noise_typ,image):
    # Switch to determine which type of noise to be added
    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        #var = 0.1
       #sigma = var**0.5
        gauss = np.random.normal(mean,10,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        #amount = 0.004
        amount = 0.25
        out = image
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy

##########################################################################
## Function Name: Main
## Function Desc.: entry point of program
##########################################################################
def main():
    Q1() # calls operations needed for question 1
    Q2() # calls methods to handle question 2
    ResidualImages() # calls method to perform operations needed for Q3
    # YES WE CAN USE ENTROPY FOR THRESHOLDING
    Entropy() # calls methods to handle question 4

    x = np.reshape(ProfGoss(5, 1.6),(1,5))

    xtran = np.reshape(x,(5,1))
    res = np.dot(xtran, x)
    print res
    return

# initializes the program
main()