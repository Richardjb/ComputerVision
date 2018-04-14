from PIL import Image
import numpy as np
from scipy.misc import imread
import scipy.ndimage
import cv2
from copy import copy, deepcopy

from math import pi, sqrt, exp
import math

import matplotlib.pyplot as plt

##########################################################################
## Function Name: Q1
## Function Desc.: Operations needed for question 1
##########################################################################
def Q1():
    # NOTE: I2 is not used
    # imports image and forces it to flatten and become grayscale
    I2 = np.asarray(Image.open("Input.jpg").convert('L'))

    # changes the  print settings to show all values in array
    np.set_printoptions(threshold='nan')

    I = Image.open('input2.jpg')
    kInfo = imread('input2.jpg')
    im = CannyEdgeDetection(I,.7,3)
    cv2.imshow("image in q1", im)
    return

##########################################################################
## Function Name: CannyEdgeDetection
## Function Desc.: Performs my implementation of Canny Edge Detection
## Function Arguments: I = PIL image, sigma = gaus arg, maskLength = len
## Function Return: image as np.array
##########################################################################
def CannyEdgeDetection(I, sigma, maskLength):
    # used to get image dimensions
    size = np.array(I).shape

    # specifies image width and height
    width  = size[1]
    height = size[0]

    # creates 1d gaussian of the passed in Length
    G = ProfGoss(maskLength,sigma)
    # creates 1D Gaussian masks for the derivitive of the function in the x and y directions respectively
    # Note* the same derivative function will be used for dx && dy because they are essentially the same
    # but treated different in orientation only
    Gx = gausXDeriv(maskLength, sigma)
    Gy = gausYDeriv(maskLength, sigma)

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

    # determines the gradient for suppresion and converts to degrees
    direction = np.arctan2(IPrimeY, IPrimeX) * 180 / np.pi

    # stores result of suppression to be used with thresholding
    nonMaxResult = NonMaximumSuppression(M, direction)

    # pauses cv2 to allow image to be seen
    cv2.waitKey(0)
    #cv2.imshow("binary", cv2.threshold(nonMaxResult,127,255,cv2.THRESH_BINARY))
    return nonMaxResult


##########################################################################
## Function Name: Q2
## Function Desc.: Quantitative Evaluation of Edge Detector
## Function Arguments: None
##########################################################################
def ZeroPadding(arr):
    print ""

##########################################################################
## Function Name: Entropy
## Function Desc.: Quantitative Evaluation of Edge Detector
## Function Arguments: None
##########################################################################
def Entropy():
    imageName = "Input.jpg"
    # imports image and forces it to flatten and become grayscale
    #image = np.asarray(Image.open(imageName).convert('L'))
    image = imread(imageName)
    height,width = image.shape
    numPx = width * height
    # array to hold entropy
    histArray = []
    valArray = []
    # indexes relate to pixel values, set to 0
    for i in range (0,255):
        histArray.append(0.0)
        valArray.append(i)
    # computes the amount of each pixel value and saves to array
    for j in range (0,height):
        for i in range(0,width):
            index = image[j][i]
            histArray[index] += 1

    # array that holds pixel density probabilities
    probArray = deepcopy(histArray)
    for i in range (0,len(histArray)):
        probArray[i] /= numPx

    #creates histogram and displays
    plt.hist(histArray,valArray, width=1)
    plt.show()

    # Store classes A and B of probability
    A = []
    B = []
    #pt+1/1-pt
    # loop to calculate p_t in A and B
    for k in range (0, 255):
        pt = 0.0
        # summation to calculate pt
        for q in range (0, k):
            pt += probArray[q]
        # prevents divide by 0
        if (pt == 0):
            A.append(0)
        else:
            A.append(float(probArray[k])/pt)
        # prevents out of array index
        if (k == 254):
            continue
        B.append(float(probArray[k + 1])/(1.0 - pt))

    print A
    print B

    HA = 0.0
    HB = 0.0
    # H(A)
    for i in A:
        it = -1 * i * np.log(i)
        #print it
        HA += it
    # H(B)
    for i in B:
        it = -1 * i * np.log(i)
        #print it
        HB += it
    # total entropy
    HT = HA + HB
    print HT
    return
    hold = 0
    holdIndex = 0;
    for j in range(0, len(HT)):
        if (HT[j] > hold):
            hold = HT[j]
            holdIndex = j

    print j
    #print HA
    #print probArray

    #print histArray

##########################################################################
## Function Name: Q2
## Function Desc.: Quantitative Evaluation of Edge Detector
## Function Arguments: None
##########################################################################
def Q2():

    name1 = "input_image.jpg"
    name1 = "input.jpg"
    name2 = "input_image2.jpg"
    # reads in test images and converts to grayscale
    testImageA = Image.open(name1).convert('L')
    testImageB = Image.open(name2).convert('L')
    testImageC = Image.open("input_image.jpg").convert('L')
    # reads in edge map for correlating images as grayscale
    testImageAEdge = np.array(Image.open("output_image.png").convert('L'))
    testImageBEdge = np.array(Image.open("output_image2.png").convert('L'))
    testImageCEdge = np.array(Image.open("output_image.png").convert('L'))

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
    testImageCInfo = imread("input_image.jpg").shape
    heightC = testImageCInfo[0]
    widthC = testImageCInfo[1]
    totalC = widthC * heightC

    '''
    # _ prevents it from being a tuple, converts image to binary
    _ ,x = cv2.threshold(nonMaxResult,7,1,cv2.THRESH_BINARY)
    print x
    '''
    # to be used with gaussian and derivative calculations best sigma is .299
    sigma = .501
    # length of masks
    maskLength = 3
    # TODO: perform canny on all images and save resulting edge as binary
    A = CannyEdgeDetection(testImageA, sigma, maskLength)
    #B = CannyEdgeDetection(testImageB, sigma, maskLength)
    #C = CannyEdgeDetection(testImageC, sigma, maskLength)

    cv2.imshow("A", A)
    cv2.waitKey(0)

    if (False):
        cv2.imshow("origA", A)
        _ ,binA = cv2.threshold(A,7,1,cv2.THRESH_BINARY)
        _ , testABin = cv2.threshold(testImageAEdge,10,1,cv2.THRESH_BINARY)
        print binA
        print testABin
        QEED(binA,testABin,widthA,heightA)
        cv2.imshow("A", binA)
        #_ ,binB = cv2.threshold(B,30,1,cv2.THRESH_BINARY)
        #cv2.imshow("B", binB)
        #_ ,binC = cv2.threshold(C,7,1,cv2.THRESH_BINARY)
        cv2.waitKey(0)


    '''
    # calculates TP for A, B, C
    countA = 0
    countB = 0
    countC = 0
    for i in range (0, heightA):
        for j in range (0, widthA):
            if (binA[i][j] == 1 and binA[i][j] == testImageAEdge[i][j]):
                countA += 1
            if (binB[i][j] == 1 and binB[i][j] == testImageBEdge[i][j]):
                countB += 1
            if (binC[i][j] == 1 and binC[i][j] == testImageCEdge[i][j]):
                countC += 1
    TP_A = countA/totalA
    TP_B = countB/totalB
    TP_C = countC/totalC

# calculates TN for A, B, C
    countA = 0
    countB = 0
    countC = 0
    for i in range (0, heightA):
        for j in range (0, widthA):
            if (binA[i][j] == 0 and binA[i][j] == testImageAEdge[i][j]):
                countA += 1
            if (binB[i][j] == 0 and binB[i][j] == testImageBEdge[i][j]):
                countB += 1
            if (binC[i][j] == 0 and binC[i][j] == testImageCEdge[i][j]):
                countC += 1
    TN_A = countA/totalA
    TN_B = countB/totalB
    TN_C = countC/totalC

    # calculates FP for A, B, C
    countA = 0
    countB = 0
    countC = 0
    for i in range (0, heightA):
        for j in range (0, widthA):
            if (binA[i][j] == 1 and binA[i][j] != testImageAEdge[i][j]):
                countA += 1
            if (binB[i][j] == 1 and binB[i][j] != testImageBEdge[i][j]):
                countB += 1
            if (binC[i][j] == 1 and binC[i][j] != testImageCEdge[i][j]):
                countC += 1
    FP_A = countA/totalA
    FP_B = countB/totalB
    FP_C = countC/totalC

    # calculates FN for A, B, C
    countA = 0
    countB = 0
    countC = 0
    for i in range (0, heightA):
        for j in range (0, widthA):
            if (binA[i][j] == 0 and binA[i][j] != testImageAEdge[i][j]):
                countA += 1
            if (binB[i][j] == 0 and binB[i][j] != testImageBEdge[i][j]):
                countB += 1
            if (binC[i][j] == 0 and binC[i][j] != testImageCEdge[i][j]):
                countC += 1
    FN_A = countA/totalA
    FN_B = countB/totalB
    FN_C = countC/totalC


    SensitivityA = TP_A/(TP_A + FN_A)
    SensitivityB = TP_B/(TP_B + FN_B)
    SensitivityC = TP_C/(TP_C + FN_C)

    SpecificityA = TN_A/(TN_A + FP_A)
    SpecificityB = TN_B/(TN_B + FP_B)
    SpecificityC = TN_C/(TN_C + FP_C)

    PrecisionA = TP_A/(TP_A + FP_A)
    PrecisionB = TP_B/(TP_B + FP_B)
    PrecisionC = TP_C/(TP_C + FP_C)

    NegativePredictiveValueA = TN_A / (TN_A + FN_A)
    NegativePredictiveValueB = TN_B / (TN_B + FN_B)
    NegativePredictiveValueC = TN_C / (TN_C + FN_C)

    FallOutA = FP_A / (FP_A + TN_A)
    FallOutB = FP_B / (FP_B + TN_B)
    FallOutC = FP_C / (FP_C + TN_C)

    FNRA = FN_A / (FN_A)
    FNRB = FN_B / (FN_B)
    FNRC = FN_C / (FN_C)
    '''

def QEED (binA,testImageAEdge, w, h):
    # area for rect or num px.
    totalA = w * h

    # calculates TP for A
    countA = 0
    for i in range (0, h):
        for j in range (0, w):
            if (binA[i][j] == 1 and binA[i][j] == testImageAEdge[i][j]):
                countA += 1

    TP_A = countA/totalA

    #calculates TN for A
    countA = 0

    for i in range (0, h):
        for j in range (0, w):
            if (binA[i][j] == 0 and binA[i][j] == testImageAEdge[i][j]):
                countA += 1

    TN_A = countA/totalA

    # calculates FP for A
    countA = 0

    for i in range (0, h):
        for j in range (0, w):
            if (binA[i][j] == 1 and binA[i][j] != testImageAEdge[i][j]):
                countA += 1
    FP_A = countA/totalA


    # calculates FN for A, B, C
    countA = 0

    for i in range (0, h):
        for j in range (0, w):
            if (binA[i][j] == 0 and binA[i][j] != testImageAEdge[i][j]):
                countA += 1

    FN_A = countA/totalA

    print TP_A
    print TN_A
    print FP_A
    print FN_A

    SensitivityA = TP_A/(TP_A + FN_A)

    SpecificityA = TN_A/(TN_A + FP_A)

    PrecisionA = TP_A/(TP_A + FP_A)

    NegativePredictiveValueA = TN_A / (TN_A + FN_A)

    FallOutA = FP_A / (FP_A + TN_A)

    FNRA = FN_A / (FN_A)

    FDRA = FP_A/(FP_A + TP_A)

    AccuracyA = (TP_A + TN_A)/(TP_A + FN_A + TN_A + FP_A)

    F_ScoreA = (2 * TP_A) / ( (2 * TP_A) + FP_A + FN_A)

    MCCA = ((TP_A * TN_A) - (FP_A * FN_A)) / \
           math.sqrt((TP_A + FP_A) * (TP_A + FN_A) * (TN_A + FP_A) * (TN_A + FN_A))

    return (SensitivityA, SpecificityA, PrecisionA, NegativePredictiveValueA, FallOutA, FNRA,
            FDRA, AccuracyA, F_ScoreA, MCCA)



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
    newImage = image.copy()
    newImagePixels = newImage.load()
    pixels = image.load()

    # loop for width then length
    for i in range (0, h):
        for j in range (0, w):
            sum = 0.0
            count = 0
            # used to calculate offset for array
            for k in range (-len(mask)/2 + 1, len(mask)/2 + 1):
                nj = j + k
                # in the event offset is out of bounds
                if (nj < 0 or nj >= w):
                    continue
                count += mask[k + len(mask)/2]
                #print mask[k + len(mask)/2]
                # performs array index * mask index and adds to sum
                sum += mask[k + len(mask)/2] * pixels[nj,i]
                # prevents divide by zero error
                if count != 0:
                    sum /= count
                else:
                    continue
            # stores value in image
            newImagePixels[j,i] = sum
    return newImage

##########################################################################
## Function Name: ProfConvolveY
## Function Desc.: Convolves image in Y direction and returns result
## Function Arguments: image = Pil, mask = 1d mask, w = width, h = height
##########################################################################
def ProfConvolveY(image, mask, w, h):
    newImage = image.copy()
    newImagePixels = newImage.load()
    pixels = image.load()

     #formatted pixel[width,height]
    for j in range (0, w):
        for i in range (0, h):
            sum = 0
            count = 0
            for k in range (-len(mask)/2 + 1, len(mask)/2 + 1):
                nj = i + k
                if (nj < 0 or nj >= h):
                    continue
                count += mask[k + len(mask)/2]
                sum += mask[k + len(mask)/2] * pixels[j,nj]
                if count != 0:
                    sum /= count
                else:
                    continue
            newImagePixels[j,i] = sum

    return newImage


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

    # since all angles have been rounded it is easy to switch between possibilities
    # check pixel along gradient direction to determine if it is local maximum
    # NOTE: the edge is perpendicular to gradient
    for y in range (0, height):
        for x in range (0, width):
            #TODO: fix 112.5
            if (dirCpy[y][x] == 135):
                if ((y + 1 < height and x + 1 < width) and (y - 1 >= 0 and x - 1 >= 0)):
                    if (magCpy[y][x] > magCpy[y + 1][x - 1] and magCpy[y][x] > magCpy[y - 1][x + 1]):
                        magCpy[y + 1][x - 1] = 0
                        magCpy[y - 1][x + 1] = 0
                        #edge[y][x] = 0
            if (dirCpy[y][x] == 112.5):
                if (y + 1 < height and y - 1 >= 0):
                    if (magCpy[y][x] > magCpy[y + 1][x] and magCpy[y][x] > magCpy[y - 1][x]):
                        magCpy[y + 1][x - 2] = 0
                        magCpy[y - 1][x + 2] = 0
                        #edge[y][x] = 0
            if (dirCpy[y][x] == 90):
                if (y + 1 < height and y - 1 >= 0):
                    if (magCpy[y][x] > magCpy[y + 1][x] and magCpy[y][x] > magCpy[y - 1][x]):
                        magCpy[y + 1][x] = 0
                        magCpy[y - 1][x] = 0
                        #edge[y][x] = 0
            if (dirCpy[y][x] == 67.5):
                if ((y + 2 < height and x + 1 < width) and (y - 2 >= 0 and x - 1 >= 0)):
                    if (magCpy[y][x] > magCpy[y + 2][x + 1] and magCpy[y][x] > magCpy[y - 2][x + 1]):
                        magCpy[y + 2][x + 1] = 0
                        magCpy[y - 2][x + 1] = 0
                        #edge[y][x] = 0
            if (dirCpy[y][x] == 45):
                if ((y + 1 < height and x + 1 < width) and (y - 1 >= 0 and x - 1 >= 0)):
                    if (magCpy[y][x] > magCpy[y + 1][x + 1] and magCpy[y][x] > magCpy[y - 1][x - 1]):
                        magCpy[y + 1][x + 1] = 0
                        magCpy[y - 1][x + 1] = 0
                        #edge[y][x] = 0
            if (dirCpy[y][x] == 22.5):
                if ((y + 1 < height and x + 2 < width) and ( y - 1 >= 0 and x - 2 >= 0)):
                    if (magCpy[y][x] > magCpy[y + 1][x + 2] and magCpy[y][x] > magCpy[y - 1][x - 2]):
                        magCpy[y + 1][x + 2] = 0
                        magCpy[y - 1][x - 2] = 0
                        #edge[y][x] = 0
            if (dirCpy[y][x] == 0):
                if (x + 1 < width and x - 1 >= 0):
                    if (magCpy[y][x] > magCpy[y][x + 1] and magCpy[y][x] > magCpy[y][x - 1]):
                        magCpy[y][x + 1] = 0
                        magCpy[y][x - 1] = 0
    return edge


##########################################################################
## Function Name: ConvolveXDir
## Function Desc.: convolves image in the x direction using arg
## image and mask
##########################################################################
def ConvvolveXDir (img, msk):
    # creates copy of original img
    Canv = deepcopy(img)
    Canv2 = deepcopy(Canv)
    # creates copy of mask
    mask = msk
    # obtains img dimensions
    size = img.shape # formatted (Height, Width)
    # sets width to image width
    width = size[1]
    # sets height to image height
    height = size[0]
    mskLen = len(mask)
    hold = -1 * (int(mskLen/2) + 1)
    holdReset = hold

    # loop to test masks of varying lengths
    for y in range (0, height - 1):
        for x in range (int(mskLen/2 + 1), width - 1):
            sum = 0.0
            hold = holdReset

            for k in range (0, len(mask)):
                #print ("Hold = %d" %hold)
                if ((x + hold) < width):
                    sum += mask[k] * img[y][x + hold]
                hold = hold + 1
                #print hold + x
            Canv2[y][x] = sum / len(mask)
    #return Canv2

    #nested loop iterates over rows then columns
    for y in range (0, height):
        for x in range (1, width - 1):
            # floats used to caclulate convolution
            sum1 = 0.0
            # applies mask to first 2 pixels
            sum1 += (Canv[y][x - 1] * mask[0]) + (Canv[y][x] * mask[1])
            sum2 = 0.0

            # logic created in event that there is a pixel out of bounds '0 padding'
            if (x + 1 >= width):
                sum2 += 0 * mask[2]
            else:
                sum2 += Canv[y][x + 1] * mask[2]

            Canv[y][x] = (sum1 + sum2) / len(mask)

    return Canv

##########################################################################
## Function Name: ConvolveYDir
## Function Desc.: convolves image in the x direction using arg
## image and mask
##########################################################################
def ConvvolveYDir (img, msk):
    # creates copy of original img
    CanvY = deepcopy(img)
    # creates copy of mask
    mask = msk
    # obtains img dimensions
    size = img.shape # formatted (Height, Width)
    # sets width to image width
    width = size[1]
    # sets height to image height
    height = size[0]


    #nested loop iterates over rows then columns
    for x in range (0, width ):
        for y in range (1, height):
            # floats used to caclulate convolution
            sum1 = 0.0
            # applies mask to first 2 pixels
            sum1 += (CanvY[y - 1][x] * mask[0]) + (CanvY[y][x] * mask[1])
            sum2 = 0.0

            # logic created in event that there is a pixel out of bounds '0 padding'
            if (y + 1 >= height):
                sum2 += 0 * mask[2]
            else:
                sum2 += CanvY[y + 1][x] * mask[2]

            CanvY[y][x] = (sum1 + sum2) / len (mask)

    return CanvY



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
## Function Name: Main
## Function Desc.: entry point of program
##########################################################################
def main():
    #Q1()# calls operations needed for question 1
    #Q2()
    Entropy()
    return

# initializes the program
main()