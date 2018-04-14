#NOTE: wikipedia, matlab and python previous implementations were used as guidance for this project

import cv2
import numpy as np
import Image
from copy import copy, deepcopy
from matplotlib import pyplot as plt
from LucasKanadeMeth1 import LucasKanadeMet1
#from LKNonPyramid import LKanade
from LKPyramid import LKanadePy
from myLK import MyLK
#from OpticalFLowFromCV2 import SampleCode


#####################################################################
## Function name: main
## Function Desc: initializes program
#####################################################################
def main():

    # definition of parameters to be used for tracking
    size_template_x = 200
    size_template_y = 200
    size_bins = 1
    n_bins = 256
    percentage_active = 20
    n_max_iters = 20
    epsilon = 0.01
    interp = 1
    n_ctrl_pts_x = 2
    n_ctrl_pts_y = 2
    lambda_tps = 0
    # dont know what these are for
    n_ctrl_pts_x_illum = 2
    n_ctrl_pts_y_illum = 2
    # coords of feature(s) to track
    coords = np.zeros(shape=(1,2))

    # loads source and destination images
    #bBallSource = cv2.imread("basketball1.png")
    #cpy = deepcopy(bBallSource)

    #bBallTarget = cv2.imread("basketball2.png")
    bBallSource = "basketball1.png"
    bBallTarget = "basketball2.png"
    bb1 = cv2.imread(bBallSource)
    bb2 = cv2.imread(bBallTarget)

    groveSource = "grove1.png"
    groveTarget = "grove2.png"
    gg1 = cv2.imread(groveSource)
    gg2 = cv2.imread(groveTarget)

    teddySource = "teddy1.png"
    teddyTarget = "teddy2.png"
    tt1 = cv2.imread(teddySource)
    tt2 = cv2.imread(teddyTarget)

    LCM1 = LucasKanadeMet1(bb1,bb2)
    LCM1 = LucasKanadeMet1(gg1,gg2)
    LCM1 = LucasKanadeMet1(tt1,tt2)

    LKPy = LKanadePy(bb1, bb2, 3, 5, 5)
    LKPy = LKanadePy(gg1, gg2, 3, 5, 5)
    LKPy = LKanadePy(tt1, tt2, 3, 5, 5)
    #LKWithoutPy = LKanade(bBallSource,bBallTarget, 40)
    #LKWithoutPy = MyLK()
    #lk = SampleCode()
    print ("Completed Lucas-Kanade")

    # NOTE: return is placed here to force stop other algorithm from being deployed
    return

    dst = 3;
    gray = cv2.cvtColor(np.array(bBallSource),cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)

    # displays the good features on the source image
    for i in corners:
        x,y = i.ravel()
        cv2.circle(bBallSource,(x,y),3,255,-1)

    gray2 = cv2.cvtColor(bBallTarget,cv2.COLOR_BGR2GRAY)
    corners2 = cv2.goodFeaturesToTrack(gray2,25,0.01,10)
    # displays the good features on the destination image
    for i in corners2:
        x,y = i.ravel()
        cv2.circle(bBallTarget,(x,y),3,255,-1)

    #plt.imshow(bBallSource),plt.show
    #plt.imshow(bBallTarget),plt.show
    cv2.imshow("src", bBallSource)
    cv2.imshow("dest", bBallTarget)
    cv2.waitKey(0)
    return
    print corners
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # threshold for an optimal value, varies depending on image
    bBallSource[dst>0.07*dst.max()]=[0,0,255]
    cv2.imshow('dst',bBallSource)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

    # create array of zeros for template
    Template = np.zeros(shape=(size_template_x,size_template_y),dtype=np.uint8)
    #Template = [[0 for x in range(size_template_x)] for x in range(size_template_y)]
    ICur = deepcopy(bBallSource)

    # gives values to template, passed/changed by reference
    Template = DefineTemplate(Template, ICur, coords)

    cv2.imshow("Template before naya", Template)
    # initializes naya structure
    # grayscale assumed
    tracker = naya(size_template_x, size_template_y, n_bins,
                   size_bins,n_max_iters,epsilon,interp=1)
    #
    # begin of tracking system
    clickX, clickY = 300,200
    # tracker parameters
    parameters = []
    parameters.append(0);parameters.append(0);parameters.append(clickX);parameters.append(clickY)

    # to be used as parameters, possibly insert tupple
    #for i in corners:
    #    x,y = i.ravel()
    #    cv2.circle(cpy,(x,y),3,255,-1)
    count = 5
    currentlyStart = True
    while(True):
        # begin engine that does lucas -kanade tracker
        tracker.RunEngine(ICur, 0, Template, 0, parameters)
        # display changes
        tracker.DisplayChange(ICur,1)
        # swaps between images
        if (count == 0):
            if(currentlyStart):
                ICur = deepcopy(bBallTarget)
            else:
                ICur = deepcopy(bBallSource)
            count = 5
    # Shows  locations of good points to track
    # Will change with edge detector as they are better for tracking
    #cv2.imshow("source1", bBallSource)
    #plt.imshow(cpy), plt.show()

    cv2.waitKey(0)



#########################################################################
# BELOW IS DEPRECATED ALL OPERATIONS ARE NOW HANDLED IN SEPARATE CLASSES
#########################################################################


#####################################################################
## Function name: DefineTemplate
## Function Desc: creates and returns template
#####################################################################
def DefineTemplate(Template, ICur, coords):
    print ICur[0][0]
    print Template[0][0]
    rows, cols = Template.shape
    #rows, cols = Template.__len__(), Template.__len__()
    # coords = location of feature to track
    # ICur = current image loaded, in grayscale
    #coords[0][0] = 100
    #coords[0][1] = 100

    for i in range (0,rows):
        for j in range (0, cols):

            #print ("i: %d\tj: %d" %(i,j))
            arg1 = np.int(coords[0][1])
            arg2 = np.int(rows/2+i)
            arg3 = np.int(coords[0][0])
            arg4 = np.int(cols/2+j)
            # use mod to make sure in bounds
            #Template[i][j] = ICur[(arg1 - arg2)%cols][(arg3 - arg4) % rows]
            Template[i][j] = ICur[(arg1 - arg2)][(arg3 - arg4)]

    M = cv2.getRotationMatrix2D((cols/2,rows/2), 180,1)
    dst = cv2.warpAffine(Template,M,(cols,rows))
    #cv2.imshow("rotation", Template)

    np.set_printoptions(threshold=np.nan)
    return dst



# class that will perform all of the operations
class naya:
    well = 8
    def __init__(self, size_template_x=5, size_template_y=5, n_bins=5,
                 size_bins=5, n_max_iters=100,epsilon=100, interp=100):
        # initialize all of the variables within class with passed in values
        self.size_template_x = size_template_x
        self.size_template_y = size_template_y
        self.epsilon = epsilon
        self.n_bins = n_bins
        self.size_bins = size_bins
        self.n_max_iters = n_max_iters
        self.interp = interp

        # Activel pixel stuff
        self.visited_r = []
        self.visited_g = []
        # for use as reference pixels
        self.std_pixel_list = []

        self.iters = 0
        # Reference pixel list
        for i in range (0, self.size_template_x * self.size_template_y):
            self.std_pixel_list.append(i)

        # rest of allocations
        self.dummy_mapx = np.zeros((size_template_y,size_template_x),dtype=np.uint8)
        self.dummy_mapy = np.zeros((size_template_y,size_template_x),dtype=np.uint8)
        self.Hess = np.zeros((2,2),dtype=np.uint8)
        self.delta = np.zeros(shape=(1,2),dtype=np.uint8)
        self.Mask = np.zeros((size_template_y,size_template_x),dtype=np.uint8)
        # set mask to 255
        for i in range (0, size_template_x):
            for j in range(0, size_template_y):
                self.Mask[j][i] = 255

        # grayscale params.
        self.SetGrayscaleParams()

        # joint histogram
        self.correction = []
        self.p_joint = []
        # populates with floats
        for i in range(0, n_bins):
            self.correction.append(0.0)
        for i in range(0, n_bins * n_bins):
            self.p_joint.append(0.0)

    # initializes params. for grayscale items
    def SetGrayscaleParams(self):
        self.dif = np.zeros((self.size_template_y,self.size_template_x),dtype=np.uint8)
        self.SD = np.zeros((self.size_template_y,self.size_template_x),dtype=np.float32)
        self.Template_comp = np.zeros((self.size_template_y,self.size_template_x),dtype=np.uint8)

        self.gradx = np.zeros((self.size_template_y,self.size_template_x),dtype=np.uint8)
        self.grady = np.zeros((self.size_template_y,self.size_template_x),dtype=np.uint8)
        self.gradx_tmplt = np.zeros((self.size_template_y,self.size_template_x),dtype=np.uint8)
        self.grady_tmplt = np.zeros((self.size_template_y,self.size_template_x),dtype=np.uint8)
        self.current_warp = np.zeros((self.size_template_y,self.size_template_x),dtype=np.uint8)

        self.expected = []
        for i in range (0, self.n_bins):
            self.expected.append(0.0)

    # begins entire process
    def RunEngine(self, ICur, Mask_roi, Template, Mask_template, parameters):
        flag_tracking = 1;

        # taking in input arguments
        self.ICur = ICur
        self.Mask_roi = Mask_roi
        self.Mask_template = Mask_template
        self.Template = Template
        self.parameters = parameters

        self.flag_tracking = 1

        # computing expected TempLate
        self.ComputeExpectedImg()

        # Are we using masks?
        if (self.Mask_roi == 0 or self.Mask_template == 0):
            using_masks = 0
        else:
            using_masks = 1

        # actual lucas-kanade ops
        for i in range(0, self.n_max_iters):
            # computing mapped positions in parallel
            self.WarpDOF()

            # Computes image gradients
            self.WarpGrad()

            # computes occlusion map
            if (using_masks):
                self.OcclusionMap()

            # Mounts Jacobian
            self.MountJacobianDOFGray()

            # Updates parameters
            if (self.UpdateDOF()):
                break

            if (self.ComputeJointHistogramGray()):
                self.flag_tracking = 0

    def OcclusionMap(self):
        print("never used")

    def WarpGrad(self):
        self.gradx = cv2.Sobel(self.current_warp,cv2.CV_32F, 1,0)
        self.grady = cv2.Sobel(self.current_warp,cv2.CV_32F, 0,1)

    def MountJacobianDOFGray(self):

        offx = np.ceil(np.double(self.size_template_x/2))
        offy = np.ceil(np.double(self.size_template_y/2))

        k = 0

        for i1 in range (0, self.size_template_y):
            for j1 in range (0, self.size_template_x):
                if (self.Mask[j1][i1] != 0):
                    # gradients
                    sum_gradx = np.float(self.gradx[j1][i1]) + np.float(self.gradx_tmplt[j1][i1])
                    sum_grady = np.float(self.grady[j1][i1]) + np.float(self.grady_tmplt[j1][i1])
                    # img difference
                    if (k < self.size_template_x):
                        self.dif[0][k] = np.float(self.current_warp[j1][i1] - self.Template_comp[j1][i1])

                    # gradient
                    if (k < self.size_template_x):
                        self.SD[0][k] = sum_gradx
                        self.SD[k][0] = sum_grady
                else:
                    #image difference
                    self.dif[0][k] = 0

                    # gradient
                    self.SD[0][k] = 0
                    self.SD[1][k] = 0

                k = k+1



    def UpdateDOF(self):
        sum = 0

        # ESM update
        SDcpy = cv2.transpose(self.SD)

        param1 = cv2.invert(np.multiply(SDcpy,self.SD))
        param2 = SDcpy*self.dif

        paramProduct = np.dot(np.asarray(param1[1], dtype=float) , np.asarray(param2,dtype=float))
        delta = 2*(paramProduct)

        # update
        self.parameters[2] -= delta[0][0]; sum+= np.fabs(delta[0][0])
        self.parameters[2] -= delta[1][0]; sum+= np.fabs(delta[1][0])


        return sum < self.epsilon

    def ComputeJointHistogramGray(self):
        print("To be implemented, NOT NEEDED")


    def WarpDOF(self):
        offx = int(np.ceil(float(self.size_template_x/2)))
        offy = int(np.ceil(float(self.size_template_y/2)))

        # Multiplying matrices
        for i in range (-offy,offy):
            for j in range (-offx, offx):
                self.dummy_mapx[j+offx][i+offy] = np.float32(j) + self.parameters[2]
                self.dummy_mapy[j+offx][i+offy] = np.float32(i) + self.parameters[3]


        #print self.ICur.type()
        # Remapping
        #cv2.remap(self.ICur, self.current_warp, self.dummy_mapx, self.dummy_mapy)
        #NOTE: would only work when converted to float32
        self.current_warp = np.uint8(cv2.remap(np.float32(self.ICur), np.float32(self.dummy_mapx),
                                               np.float32(self.dummy_mapy), cv2.INTER_LINEAR))

        M = cv2.getRotationMatrix2D((self.size_template_y/2,self.size_template_x/2), 270,1)
        self.current_warp = cv2.warpAffine(self.current_warp,M,(self.size_template_y,self.size_template_x))
        cv2.imshow("FIXING WARP", self.current_warp)
        # is using masks
        if (False):
            print ("not using masks")

    def ComputeExpectedImg(self):
        # Calculates intensity value to be added to each
        #  intensity value in reference image
        for i in range(0, self.n_bins):
            self.correction[i] = self.size_bins*(self.expected[i] - i)

        # correcting template
        for v in range(0, self.size_template_y):
            for u in range (0, self.size_template_x):
                #TODO: add float cast
                self.Template_comp[v][u] = self.Template[v][u] + \
                                           np.uint(np.round(self.correction[np.uint(np.round(self.Template[v][u]/self.size_bins))]))


        # Re-computing Gradient
        self.gradx_tmplt = cv2.Sobel(np.uint8(self.Template_comp), cv2.CV_32F,1,0, ksize=3)
        self.grady_tmplt = cv2.Sobel(np.uint8(self.Template_comp), cv2.CV_32F,0,1, ksize=3)



    def DisplayChange(self, ICur, delay):
        x1 = np.int(-self.size_template_x/2 + self.parameters[2])
        x2 = np.int(self.size_template_x/2 + self.parameters[2])
        y1 = np.int(-self.size_template_y/2 + self.parameters[3])
        y2 = np.int(self.size_template_y/2 + self.parameters[3])
        topLeftCorner = (x1,y2)
        bottomRightCorner = (x2,y1)
        cv2.rectangle(ICur,topLeftCorner, bottomRightCorner, (0,255,0), thickness=3)


        if (self.current_warp.size != 0):
            cv2.imshow("Current Image", self.current_warp)
        if (self.current_warp.size != 0):
            cv2.imshow("ICur", np.uint8(self.ICur))
        if (self.current_warp.size != 0):
            cv2.imshow("Template_comp", np.uint8(self.Template_comp))

        return cv2.waitKey(delay)

# Initializes program
main()