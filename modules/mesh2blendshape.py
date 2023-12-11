import argparse
import time
from math import cos, sin, pi
from os import path
import importlib
import cv2
import numpy as np
from PIL import Image
from PIL import ImageTk
import tkinter as tk
from face_module.LDT.lib import BMManager

def Mesh2Blendshape(path):
    print("******Load given Blendshape meshes******")
    BM = BMManager.BMMng('face_module/LDT/data/faceXmodel_bfm/', path)
    BM.LoadBS()
    print(len(BM.BlendShapes), "Base blendshapes are loaded")

    # Load Avatar
    print("******Load Avatar******")
    BM.LoadAvatar()

    # Rescale
    print("******Rescale Avatar******")
    BM.RescaleAvatar()

    # Triangle Correspondence
    print("******Triangle Correspondence******")
    BM.TriangleCorrespondence(0)

    # Deformation Transfer
    print("******Deformation transfer******")
    BM.DeformationTransfer()

    # End
    print("LDT Complete.")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Make Blendshape models of given 3D Avatar')
    # parser.add_argument('--ldt_path', default="face_module/LDT/Inputs/Mario.ply", type=str, help='path for 3D Avatar')
    # args = parser.parse_args()
    path = ""
    Mesh2Blendshape(path)