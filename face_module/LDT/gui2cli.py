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
from .lib import BMManager

base_dir = "/home/ubuntu/3d_temp"
def LDT(path):
    print("******Load given Blendshape meshes******")
    BM = BMManager.BMMng(f'{base_dir}/BlendShapeMaker/data/faceXmodel_head_tri/', path)
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
    parser = argparse.ArgumentParser(description='Make Blendshape models of given 3D Avatar')
    parser.add_argument('--ldt_path', default=f"{base_dir}/BlendShapeMaker/Inputs/Mario.ply", type=str, help='path for 3D Avatar')
    args = parser.parse_args()

    LDT(args.ldt_path)