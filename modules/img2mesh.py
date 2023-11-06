import time
from pathlib import Path

import torch
from loguru import logger
import trimesh

from face_module.MICA.MICA import MICA, process, to_batch, process_single


class Image2Mesh:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        start_time = time.time()
        print("Conver image to face meshes...")
        print("Loading MICA...")
        self.mica, self.faces, self.app = MICA(cfg, args)
        self.lap_time_0 = time.time()
        print(f"\033[1;3;31mLoading MICA Took... \n\t{self.lap_time_0 - start_time}s\033[0m")
        
    def __call__(self):
        with torch.no_grad():
            logger.info(f'Processing has started...')
            path = process_single(self.args, self.app, draw_bbox=False)

            name = Path(path).stem
            images, arcface = to_batch(path)
            codedict = self.mica.encode(images, arcface)
            opdict = self.mica.decode(codedict)
            meshes = opdict['pred_canonical_shape_vertices']
            code = opdict['pred_shape_code']
            lmk, lmk_faces, face_tensors = self.mica.flame.compute_landmarks(meshes) #landmark vertices 좌표, face idx도 반환하게 할수는 있는데 vertex idx는 안되는듯..? 필요한건 vertex idx긴한데

            vtx_lmk_idxs = face_tensors[lmk_faces][0][17:,0].tolist() #68개의 Facial Landmark 중 사용하는 51개, 그리고 그 중 Landmark Face 의 3개 Vertex 중 0번째 선택
            data_str = ' '.join(map(str, vtx_lmk_idxs))

            with open(f'face_module/LDT/data/landmarks/landmarks_MICA.txt', 'w') as f:
                f.write(data_str)
            
            mesh = meshes[0]
            # landmark_51 = lmk[0, 17:]
            # landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

            # dst = Path(args.o, name)
            # dst.mkdir(parents=True, exist_ok=True)
            
            
            trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=self.faces, process=False).export(f'face_module/LDT/Inputs/MICA.ply', file_type='ply', encoding='ascii')  # save in millimeters We only need ply... Save directly to LDT
            
            
            # trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
            # np.save(f'{dst}/identity', code[0].cpu().numpy())   ##### Use if we use flame, or tracker
            # np.save(f'{dst}/kpt7', landmark_7.cpu().numpy() * 1000.0)       ######Landmrak..
            # np.save(f'{dst}/kpt68', lmk.cpu().numpy() * 1000.0)           #####Landmark...

            lap_time_mica = time.time()
            print(f"\033[1;3;31mRunning MICA Took... \n\t{lap_time_mica - self.lap_time_0}s\033[0m")