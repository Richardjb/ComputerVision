__author__ = 'Richard Jean-Baptiste'
import random
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy.matlib
from mpl_toolkits.mplot3d import Axes3D






##################################################################################
#   Function Name: ProbTwo ()
#   Description  : Performs Calculations for problem two
##################################################################################
def ProbTwo():
    #recieves input from user as string
    upperBound = raw_input("Please enter an positive integer greater than one as upperBound: ")
    #converts string to integer
    upperBoundInt = int(upperBound)
    print ("Think of an integer between 1 and %d"%upperBoundInt)
    #Call Guessing function with log_2(n) complexity
    GuessGame(upperBoundInt, int(upperBoundInt/2))

##################################################################################
#   Function Name: GuessGame(higherValue, currentGuess)
#   Description  : Performs looping operation
##################################################################################
def GuessGame(higherValue, currentGuess):
    #The number guesses must be positive
    lowerValue = 1
    #There can be a max of n guesses (actually n/2)
    for i in range(higherValue):
        print "Is your number " + str(currentGuess) + "?"
        userResponse = raw_input("Input H if the number is higher than what you have in mind.\n"
                             "Input L if the number is lower than what you have in mind.\n"
                             "Input Y if the number is what you have in mind.\n")
        if (userResponse == "Y"):
            print("Your number is %d thanks for playing!"%currentGuess)
            return
        if (userResponse == "H"):
            higherValue = currentGuess - 1
            currentGuess = (higherValue + lowerValue) / 2
            continue
        if (userResponse == "L"):
            lowerValue = currentGuess + 1
            currentGuess = (higherValue + lowerValue) / 2
            continue
    print("You are not playing fair!");

##################################################################################
#   Function Name: ProbThreeA()
#   Description  : Determines Perfect Numbers less than 10,00
##################################################################################
def ProbThreeA():
    perfectNumberList =  [1]
    perfectNumberList.remove(1)
    for i in range(2, 10000):
        perfectNum = IsPerfectNumber(i)
        if (perfectNum) :
            perfectNumberList.append(i)
    for i in perfectNumberList:
        print(i)
##################################################################################
#   Function Name: IsPerfectNumber (numToCheck)
#   Description  : Determines if a number is Perfect and returns true or false
##################################################################################
def IsPerfectNumber(numToCheck):
    #used to contain list of factors
    divisorList = [0]
    divisorList.remove(0)
    sum = 0
    #loop that populates factors
    for i in range(1, numToCheck - 1):
        if ( numToCheck % i == 0):
            divisorList.append(i)

    #sums the factors
    for j in divisorList:
        sum += j

    if (sum == numToCheck):
        print divisorList
        return True
    return False

##################################################################################
#   Function Name: ProbThreeA
#   Description  : Generates random 3 x 3 matrix and determines singularity
################################################################################
def ProbThreeB():
    randomList = []
    #generates random numbers
    for i in range(9):
        # float matrix
        randomList.append(random.uniform(0,1))
        # Integer matrix
        #randomList.append(random.randint(0,1))

    #creates 3 x 3 matrix of random numbers
    randomMatrix = [[randomList[0], randomList[1], randomList[2]],  # (0,0) (0,1) (0,2) respectively
                    [randomList[3], randomList[4], randomList[5]],  # (1,0) (1,1) (1,2) respectively
                     [randomList[6], randomList[7], randomList[8]]] # (2,0) (2,1) (2,2) respectively

    #creates 9 random numbers and places them in matrix
    randomList = numpy.random.rand(3,3)
    print randomMatrix

    deter = isSingular(randomMatrix)

##################################################################################
#   Function Name: isSingular (Matrix)
#   Description  : Calculates the determinant of passed in maxtrix and determines
#                   Singularity
################################################################################
def isSingular(Matrix):

    #Creates another Matrix to find determinant
    solveMatrix = [[Matrix[0][0], Matrix[1][1], Matrix[2][0]],
                    [Matrix[0][1], Matrix[1][1], Matrix[2][1]],
                     [Matrix[0][2], Matrix[1][2], Matrix[2][2]],
                      [Matrix[0][0], Matrix[1][1], Matrix[2][0]],
                        [Matrix[0][1], Matrix[1][1], Matrix[2][1]]]
    '''
    #Creates another Matrix to find determinant
    solveMatrix = [[Matrix[0][0], Matrix[1][0], Matrix[2][0]],
                    [Matrix[0][1], Matrix[1][1], Matrix[2][1]],
                     [Matrix[0][2], Matrix[1][2], Matrix[2][2]],
                      [Matrix[0][0], Matrix[1][0], Matrix[2][0]],
                        [Matrix[0][1], Matrix[1][1], Matrix[2][1]]]
    '''
    rightDiagnalSum = 0
    rightDiagnalSumMult = 1
    leftDiagnalSumMult = 1

    # Does right diagonal
    for i in range(3):
        for j in range (3):
            rightDiagnalSumMult *= solveMatrix[i + j][j]
        rightDiagnalSum += rightDiagnalSumMult
        # resets to 1 for next diagnal
        rightDiagnalSumMult = 1

    # Does left diagonal
    for i in range(4, 1, -1):
       for j in range (3):
           leftDiagnalSumMult *= solveMatrix[i - j][j]
       rightDiagnalSum -= leftDiagnalSumMult
       leftDiagnalSumMult = 1

    #a matrix is defined as singular when the determinant is 0
    if (rightDiagnalSum != 0):
        print ("Not Singular determinant = " + str(rightDiagnalSum))
    else: print ("Matrix is Singular")
    return (rightDiagnalSum == 0)

##################################################################################
#   Function Name: ProbOne()
#   Description  : Finds the highest 4 palindromes whose sum is a palindrome
#                   Then multiplies the 4 distinct numbers together
##################################################################################
def ProbOne():
    # holds all possible palindromes in range
    palindromeList = []
    count = 0
    for i in range (10, 1000):
        if (IsPalindrome(i)):
            palindromeList.append(i)
            count += 1
            #print ("count: %d Palindrome: %d" % (count, i))

    # reverses list to find maximum numbers quicker
    palindromeList.reverse()
    # creates generator
    container = combinations(palindromeList, 4)

    highestMult = 0
    dig1 = 0
    dig2 = 0
    dig3 = 0
    dig4 = 0
    for i in container:
        # summation is a palindrome
        if (IsPalindrome(i[0] + i[1] + i[2] + i[3])):
            # stores the highest product
            if ((i[0] * i[1] * i[2] * i[3]) > highestMult):
                highestMult = i[0] * i[1] * i[2] * i[3]
                dig1 = i[0]
                dig2 = i[1]
                dig3 = i[2]
                dig4 = i[3]
    print ("%d * %d * %d * %d = %d" % (dig1, dig2, dig3, dig4, highestMult))
    print ("%d + %d + %d + %d = %d" % (dig1, dig2, dig3, dig4, dig4 + dig3 + dig2 + dig1))
##################################################################################
#   Function Name: Combinations()
#   Description  : Creates all possible combinations
##################################################################################
def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

##################################################################################
#   Function Name: IsPalindrome(num)
#   Description  : Determines if a number is a palindrome
##################################################################################
def IsPalindrome(num):
    # converts number to string
    numForward = str(num)
    # reverses number
    numBackward = numForward[::-1]
    if (numForward == numBackward):
        return True
    else: return False

##################################################################################
#   Function Name: ProbFour()
#   Description  : Does Monte Carlo Simulation
##################################################################################
def ProbFour():
    coordinate = []
    # 10 ^ 6 is a mil.
    numElements = 10 ** 6
    #numElements = 1000
    # generates 1 mil. coordinates 10 ^ 6
    for i in range(numElements):
        x = random.randint(-10,10)
        y = random.randint(-10,10)
        if (x == 0 or y == 0):
            while (x == 0 or y == 0):
                x = random.randint(-10,10)
                y = random.randint(-10,10)

        coordinate.append((x,y))
        #print coordinate

    oddCount = 0
    # (a) graph
    for i in coordinate:
        if (ValidPoint(i, 1)):
            oddCount += 1

    print ("Probability dart falls in odd-numbered region in square a = %f" % (float(oddCount) / float(numElements)))
    # resets odd count for next graph
    oddCount = 0
    # (b) graph
    for i in coordinate:
        if (ValidPoint(i, 2)):
            oddCount += 1
            continue

    print ("Probability dart falls in odd-numbered region in square b = %f" % (float(oddCount) / float(numElements)))


##################################################################################
#   Function Name: ValidPoint(pt)
#   Description  : Determines if y is in a valid area relating to x
##################################################################################
def ValidPoint(pt, graph):
    x = pt[0]
    y = pt[1]

    b = 0
    # line formula for first quadrant
    q = -1 * b + 10
    # performs calculations for square b
    if (graph == 1):
        if (x <= 0 ):
            return True

        if (x > 0 and y < 0):
            return False

        if ((-1 * x + 10) > y):
            return True

        if ((-1 * x + 10) < y):
            return False

        return True
    # performs calculations for square b
    if (graph == 2):
        if (x <= 0 and y <= 0):
            return True

        if (x < 0 and y > 0):
            return False

        if (x > 0 and y < 0):
            return False

        if ((-1 * x + 10) > y):
            return True

        if ((-1 * x + 10) < y):
            return False


##################################################################################
#   Function Name: ProbFive()
#   Description  : Draws graph and performs required calculations for problem
#                  5
##################################################################################
def ProbFive():
    # lists to store paths of each individual points
    pathOne = []
    pathTwo = []
    pathThree = []

    # used to scale triangle
    scalar = 10
    # coordinates of eql. triangle in Q1
    pathOne.append(((float(0) * scalar),float(0) * scalar)) # bottom left
    pathTwo.append(((float(4) * scalar,float(0) * scalar))) # bottom right
    pathThree.append((float(2) * scalar,float(3) * scalar)) # top

    time = 0 # used as index
    step = 1 # delta s
    traveled = 0 # total distance
    while True:
        #coordinates meet
        if ( pathOne[time] == pathTwo[time] == pathThree[time]):
            print ("they meet")
            break

        #infinite loop tracker
            print ("time is 100")
            print distance(pt1, pt2)
            print distance(pt2, pt3)
            print distance(pt1, pt3)
            break


        pt1 = pathOne[time]
        pt2 = pathTwo[time]
        pt3 = pathThree[time]

        #distance between points are 0
        if (distance(pt1, pt2) == 0 and distance(pt2, pt3) == 0):
            print "worked"
            break

        # changes step size
        if (distance(pt1,pt2) < 2):
            step = .5

        # gets next point closer to target
        pathOne.append(gapCloser(step,pt1,pt2))
        pathTwo.append(gapCloser(step,pt2,pt3))
        pathThree.append(gapCloser(step,pt3,pt1))

        traveled += distance(pathOne[0], pathOne[1])
        # adds additional point if the distance is stepsize
        if (distance(pt1, pt2) == step):
            pathOne.append(pt2)
            pathTwo.append(pt2)

        if (distance(pt2, pt3) == step):
            pathTwo.append(pt3)
            pathThree.append(pt3)
        time += 1
    #print pathOne
    #print pathTwo
    #print pathThree
    i = pathOne[0]
    j = pathOne[1]
    oneX = []
    OneY = []
    twoX = []
    twoY = []
    threeX = []
    threeY = []
    z1 = []
    z2 = []
    z3 = []

    for i in pathOne:
        oneX.append(i[0])
        OneY.append(i[1])
        z1.append(0)

    for i in pathTwo:
        twoX.append(i[0])
        twoY.append(i[1])
        z2.append(0)

    for i in pathThree:
        threeX.append(i[0])
        threeY.append(i[1])
        z3.append(0)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(oneX, OneY, z1, c='r', marker='o')
    ax.scatter(twoX, twoY, z2, c='b', marker='o')
    ax.scatter(threeX, threeY, z3, c='g', marker='o')

    print pathOne
    print pathTwo
    print pathThree

    speed = 1

    print ("The ants collide after %d seconds moving at a speed of .%d mph" %(time, speed))
    print ("They follow the paths in correlating to their color on the graph")
    print ("The total distance traveled was %.2f miles" %float(traveled))


    plt.show()

##################################################################################
#   Function Name: gapCloser (stepSize, coord, dest)
#   Description  : Shortens the distance between two points
##################################################################################
def gapCloser (stepSize, coord, dest):
    tempX = 0.0
    tempY = 0.0
    if (coord[0] < dest[0]):
        tempX = coord[0] + stepSize
    if (coord[0] > dest[0]):
        tempX = coord[0] - stepSize
    if (coord[0] == dest[0]):
        tempX = dest[0]

    if (coord[1] < dest[1]):
        tempY = coord[1] + stepSize
    if (coord[1] > dest[1]):
        tempY = coord[0] - stepSize
    if (coord[1] == dest[1]):
        tempY = dest[1]

    newCoordinate = (tempX,tempY)
    #print newCoordinate

    return newCoordinate

##################################################################################
#   Function Name: distance (p1,p2):
#   Description  : Uses the distance formula and returns distance between 2 points
##################################################################################
def distance (p1,p2):
    #distance formula
    return numpy.sqrt(numpy.square(p1[0] - p2[0]) + numpy.square(p1[1] - p2[1]))


##################################################################################
#   Function Name: Main Program
#   Description  : Programs entry point
##################################################################################
ProbOne()
ProbTwo()
ProbThreeA()
ProbThreeB()
ProbFour()
ProbFive()
