
import argparse
import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import trimesh
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from skimage.io import imread
from tqdm import tqdm

from MICA.configs.config import get_cfg_defaults
from MICA.datasets.creation.util import get_arcface_input, get_center, draw_on
from MICA.utils import util
from MICA.utils.landmark_detector import LandmarksDetector, detectors
from MICA.MICA import MICA, deterministic, process, to_batch, process_single





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MICA - Towards Metrical Reconstruction of Human Faces')
    parser.add_argument('--device', default='cuda', type=str, help='cuda')
    parser.add_argument('-i', default='input', type=str, help='Input folder with images')
    parser.add_argument('-path', default='/home/work/YAI-Summer/junwan/GPT2FACE/VITS/rdj.webp', type=str, help='Input with images')
    parser.add_argument('-o', default='output_MICA', type=str, help='Output folder')
    parser.add_argument('-a', default='MICA/demo/arcface', type=str, help='Processed images for MICA input')
    parser.add_argument('-m', default='MICA/data/pretrained/mica.tar', type=str, help='Pretrained model path')
    

    args = parser.parse_args()



    Path(args.o).mkdir(exist_ok=True, parents=True)

    cfg = get_cfg_defaults()

    deterministic(42)
    mica, faces, app = MICA(cfg, args)

    # Batch
    # with torch.no_grad():
    #     logger.info(f'Processing has started...')
    #     paths = process(args, app, draw_bbox=False)
    #     for path in tqdm(paths):
    #         name = Path(path).stem
    #         images, arcface = to_batch(path)
    #         codedict = mica.encode(images, arcface)
    #         opdict = mica.decode(codedict)
    #         meshes = opdict['pred_canonical_shape_vertices']
    #         code = opdict['pred_shape_code']
    #         lmk = mica.flame.compute_landmarks(meshes)

    #         mesh = meshes[0]
    #         landmark_51 = lmk[0, 17:]
    #         landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

    #         dst = Path(args.o, name)
    #         dst.mkdir(parents=True, exist_ok=True)
    #         trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.ply')  # save in millimeters
    #         trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
    #         np.save(f'{dst}/identity', code[0].cpu().numpy())
    #         np.save(f'{dst}/kpt7', landmark_7.cpu().numpy() * 1000.0)
    #         np.save(f'{dst}/kpt68', lmk.cpu().numpy() * 1000.0)

    #     logger.info(f'Processing finished. Results has been saved in {args.o}')


    #Single Image
    with torch.no_grad():
        logger.info(f'Processing has started...')
        path = process_single(args, app, draw_bbox=False)
        # for path in tqdm(paths):

        name = Path(path).stem
        images, arcface = to_batch(path)
        codedict = mica.encode(images, arcface)
        opdict = mica.decode(codedict)
        meshes = opdict['pred_canonical_shape_vertices']
        code = opdict['pred_shape_code']
        lmk, lmk_faces, face_tensors = mica.flame.compute_landmarks(meshes) #landmark vertices 좌표, face idx도 반환하게 할수는 있는데 vertex idx는 안되는듯..? 필요한건 vertex idx긴한데

        vtx_lmk_idxs = face_tensors[lmk_faces][0][17:,0] #68개의 Facial Landmark 중 사용하는 51개, 그리고 그 중 Landmark Face 의 3개 Vertex 중 0번째 선택

        mesh = meshes[0]
        landmark_51 = lmk[0, 17:]
        landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

        dst = Path(args.o, name)
        dst.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.ply')  # save in millimeters
        trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
        np.save(f'{dst}/identity', code[0].cpu().numpy())
        np.save(f'{dst}/kpt7', landmark_7.cpu().numpy() * 1000.0)
        np.save(f'{dst}/kpt68', lmk.cpu().numpy() * 1000.0)

        logger.info(f'Processing finished. Results has been saved in {args.o}')