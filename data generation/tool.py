from cv2 import transform
import numpy as np
import SimpleITK as sitk
from numpy.lib.function_base import copy
from copy import deepcopy


def getRotationMatrix(rx=0, ry=0, rz=0):
    rx *= np.pi / 180
    ry *= np.pi / 180
    rz *= np.pi / 180

    matx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    maty = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    matz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    return matx @ maty @ matz


class tool:
    ''' the main class to simulate the tool and apply spatial transformations on it
    As input, this class takes the 3D volume of the tool and the corresponding 3D landmark coordinates'''
    def __init__(self, tool, lnd, verbose=True):
        self.tool = tool
        self.out_tool = None
        self.out_land = None
        self.land = lnd   # 3D coordinates
        self.random = None
        self.trn_range = None
        self.rot_range = None
        self.trn = None
        self.rot=None
        self.rot_centre = None
        self.verbose = verbose
    

    
    def SetRotationCentre(self, rot_centre=None):
        ''' This function will calculate the centroid of the tool volume'''
        if rot_centre == None:
            if self.verbose:
                print("rotation centre not specified, setting it to:")
            rot_centre=np.zeros(3)
            for i in range(0,3):
                rot_centre[i] = self.tool.GetOrigin()[i] + (self.tool.GetSpacing()[i] * self.tool.GetSize()[i]) * 0.5

            self.rot_centre = rot_centre
        else:
            self.rot_centre = rot_centre

        if self.verbose:
            print("rotation centre:", self.rot_centre)


    def CentralizeTool(self):
        spacing = np.array(self.tool.GetSpacing())
        size = np.array(self.tool.GetSize())
        self.tool.SetOrigin(- size * spacing * 0.5)
        
    

    def SetTransformRange(self, trn_range=np.zeros(2), rot_range=np.zeros(2)):
        ''' Use this function when you want to set the RANGE of transformation parameters
        each time a new DRR is created, a random transformation will be applied on the tool
        within the specified RANGE'''
        if self.verbose:
            print("Setting the random transformation state to True first")
        self.random = True
        self.trn_range = trn_range
        self.rot_range = rot_range
        if self.rot_centre == None:
            self.SetRotationCentre()
        


    def SetTransformationParameters(self, trn=np.zeros(3), rot=np.zeros(3)):
        ''' Use this function when you have the list of rotation and translation parameters
        the transformations will not be applied randomly'''
        if self.verbose:
            print("Setting the random transformation state to False first")
        self.random = False
        self.trn = trn
        self.rot = rot
        if self.rot_centre == None:
            self.SetRotationCentre()



    def GetTool(self, use_reference=True):
        ''' Returns the transformed tool and the transformed 3D landmakr coordinates'''
        out_tool, affine, rx, ry, rz, trnMat = self.moveTool(use_reference)
        lanlan = np.asarray([affine.GetInverse().TransformPoint(p) for p in self.land])
        return out_tool, lanlan, rx, ry, rz, trnMat
        

    def moveTool(self, use_reference=True):
        ''' moves the tool'''
        affine = sitk.AffineTransform(3)

        if self.random:
            matrix, rx, ry, rz, trnMat= self.generateRandomPose(trn_range=self.trn_range, rot_range=self.rot_range, rot_centre=self.rot_centre)

        else: 
            matrix = self.generatePose(trn=self.trn, rot=self.rot, rot_centre=self.rot_centre)

            rx, ry, rz, trnMat = 0, 0 ,0 , 0
 

        affine.SetTranslation(matrix[0:3, 3].ravel())
        affine.SetMatrix(matrix[0:3,0:3].ravel())
        
        resampler = sitk.ResampleImageFilter()

        if use_reference:
            resampler.SetOutputDirection(self.tool.GetDirection())
            resampler.SetOutputOrigin(self.tool.GetOrigin())
            resampler.SetOutputSpacing(self.tool.GetSpacing())
            resampler.SetSize(self.tool.GetSize())

        else:
            new_direction = matrix[0:3, 0:3] @ np.reshape(self.tool.GetDirection(), (3, 3))
            new_spacing = matrix[0:3, 0:3] @ self.tool.GetSpacing()
            old_origin = np.zeros(4)
            old_origin[0:3] = self.tool.GetOrigin()
            new_origin = matrix @ old_origin
            resampler.SetOutputDirection(new_direction.ravel())
            resampler.SetOutputSpacing([1, 1, 1])
            resampler.SetOutputOrigin(new_origin[0:3])
            resampler.SetSize([700, 700, 700])


        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(affine)
        resampler.SetOutputPixelType(sitk.sitkUInt8)

        return resampler.Execute(self.tool), affine, rx, ry, rz, trnMat


    
    def generateRandomPose(self, trn_range, rot_range, rot_centre):
        
        trnMat = np.eye(4)     
        rotMat = np.eye(4) 
        cenMat = np.eye(4)
        mincenMat = np.eye(4)

        trnMat[0:3, -1] = np.random.uniform(low=trn_range[0], high=trn_range[1], size=(3,))
        rx, ry, rz = np.random.uniform(low=rot_range[0], high=rot_range[1], size=(3,))
        rotMat[0:3, 0:3] = getRotationMatrix(rx=rx, ry=ry, rz=rz)
        cenMat[0:3, -1] = rot_centre
        mincenMat[0:3, -1] = -1 * rot_centre

        return trnMat @ (cenMat @ rotMat @ mincenMat) ,rx, ry, rz, trnMat
    

    def generatePose(self, trn, rot, rot_centre):
        trnMat = np.eye(4)     
        rotMat = np.eye(4) 
        cenMat = np.eye(4)
        mincenMat = np.eye(4)

        trnMat[0:3, -1] = trn[:]
        rotMat[0:3, 0:3] = getRotationMatrix(rx=rot[0], ry=rot[1], rz=rot[2])
        cenMat[0:3, -1] = rot_centre
        mincenMat[0:3, -1] = -1 * rot_centre

        matrix = trnMat @ (cenMat @ rotMat @ mincenMat) 

        return matrix


