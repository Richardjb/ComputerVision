from FaceRec import FaceRec
from GraphCut import GraphCut

# programs start point
def main():

    # performs facial recognition for each subject
    _faceRec = FaceRec(FaceRec.USING_LBP_HISTOGRAM)
    _faceRec = FaceRec(FaceRec.USING_INTEGRAL_INTENSITIES)
    _faceRec = FaceRec(FaceRec.USING_IMAGE_INTENSITIES)
    _faceRec = FaceRec(FaceRec.USING_ALL)
    # performs graphcut for each image in graphcut directory
    graphTest = GraphCut()
    print ("Complete")
# calls main program entrypoint
main()