import time
from pathlib import Path
import os

import torch
from loguru import logger
import trimesh

from face_module.MICA.MICA import MICA, process, to_batch, process_single
from face_module.MICA.configs.config import get_cfg_defaults


class Image2Mesh:
    def __init__(self, args):
        self.cfg = get_cfg_defaults()
        self.args = args
        self.mica, self.faces, self.app = MICA(self.cfg, args)
        self.lap_time_0 = time.time()

    def __call__(self, image_path, output_filename):
        with torch.no_grad():
            logger.info(f"Processing has started...")
            path = process_single(self.args, self.app, image_path, draw_bbox=False)

            name = Path(path).stem
            images, arcface = to_batch(path)
            codedict = self.mica.encode(images, arcface)
            opdict = self.mica.decode(codedict)
            meshes = opdict["pred_canonical_shape_vertices"]
            code = opdict["pred_shape_code"]
            lmk, lmk_faces, face_tensors = self.mica.flame.compute_landmarks(
                meshes
            )  # landmark vertices 좌표, face idx도 반환하게 할수는 있는데 vertex idx는 안되는듯..? 필요한건 vertex idx긴한데

            vtx_lmk_idxs = face_tensors[lmk_faces][0][
                17:, 0
            ].tolist()  # 68개의 Facial Landmark 중 사용하는 51개, 그리고 그 중 Landmark Face 의 3개 Vertex 중 0번째 선택
            data_str = " ".join(map(str, vtx_lmk_idxs))

            with open(f"face_module/LDT/data/landmarks/landmarks_{output_filename}.txt", "w") as f:
                f.write(data_str)

            mesh = meshes[0]
            # landmark_51 = lmk[0, 17:]
            # landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

            # dst = Path(args.o, name)
            # dst.mkdir(parents=True, exist_ok=True)

            trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=self.faces, process=False).export(
                os.path.join(self.args["mica_param"]["output_path"], f"{output_filename}.ply"),
                file_type="ply",
                encoding="ascii",
            )  # save in millimeters We only need ply... Save directly to LDT

            # trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
            # np.save(f'{dst}/identity', code[0].cpu().numpy())   ##### Use if we use flame, or tracker
            # np.save(f'{dst}/kpt7', landmark_7.cpu().numpy() * 1000.0)       ######Landmrak..
            # np.save(f'{dst}/kpt68', lmk.cpu().numpy() * 1000.0)           #####Landmark...

            lap_time_mica = time.time()
