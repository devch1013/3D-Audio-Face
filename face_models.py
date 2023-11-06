import torch
import librosa
import time

from T.demo import test as SpeechToFace
from T.model import EmoTalk

# from tortoise.api import TextToSpeech, MODELS_DIR

import os
import sys

from MICA.MICA import MICA, deterministic, process, to_batch, process_single

from pathlib import Path
import torch
import trimesh
from loguru import logger

from BlendShapeMaker.gui2cli import LDT

# sys.path.append(f"{base_dir}/vits")
base_dir = "/home/ubuntu/3d_temp"
class Img2Face:
    def __init__(self, cfg, args):
        self.mica, self.faces, self.app = MICA(cfg, args)
        self.cfg = cfg
        self.args = args
        
    def __call__(self):
        with torch.no_grad():
            logger.info(f'Processing has started...')
            path = process_single(self.args, self.app, draw_bbox=False)
            # for path in tqdm(paths):

            name = Path(path).stem
            images, arcface = to_batch(path)
            codedict = self.mica.encode(images, arcface)
            opdict = self.mica.decode(codedict)
            meshes = opdict['pred_canonical_shape_vertices']
            code = opdict['pred_shape_code']
            lmk, lmk_faces, face_tensors = self.mica.flame.compute_landmarks(meshes) #landmark vertices 좌표, face idx도 반환하게 할수는 있는데 vertex idx는 안되는듯..? 필요한건 vertex idx긴한데

            vtx_lmk_idxs = face_tensors[lmk_faces][0][17:,0].tolist() #68개의 Facial Landmark 중 사용하는 51개, 그리고 그 중 Landmark Face 의 3개 Vertex 중 0번째 선택
            data_str = ' '.join(map(str, vtx_lmk_idxs))

            with open(f'{base_dir}/BlendShapeMaker/data/landmarks/landmarks_MICA.txt', 'w') as f:
                f.write(data_str)
            
            mesh = meshes[0]
            # landmark_51 = lmk[0, 17:]
            # landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

            # dst = Path(args.o, name)
            # dst.mkdir(parents=True, exist_ok=True)
            
            
            trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=self.faces, process=False).export(f'{base_dir}/BlendShapeMaker/Inputs/MICA.ply', file_type='ply', encoding='ascii')  # save in millimeters We only need ply... Save directly to LDT
            
            
            # trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
            # np.save(f'{dst}/identity', code[0].cpu().numpy())   ##### Use if we use flame, or tracker
            # np.save(f'{dst}/kpt7', landmark_7.cpu().numpy() * 1000.0)       ######Landmrak..
            # np.save(f'{dst}/kpt68', lmk.cpu().numpy() * 1000.0)           #####Landmark...

            lap_time_mica = time.time()
            # print(f"\033[1;3;31mRunning MICA Took... \n\t{lap_time_mica - lap_time_0}s\033[0m")
            
            print(self.args.ldt_path)
            LDT(self.args.ldt_path)
            lap_time_ldt = time.time()
            print(f"\033[1;3;31mRunning LDT Took... \n\t{lap_time_ldt - lap_time_mica}s\033[0m")


sys.path.append(f"{base_dir}/Emotalk")

class EmotalkFace:
    def __init__(self):
        self.args = {
            "feature_dim": 832,
            "bs_dim": 52,
            "device": "cuda",
            "batch_size": 1,
            "max_seq_len": 5000,
            "period": 30,
        }
        
        model_path = f"{base_dir}/EmoTalk_release/pretrain_model/EmoTalk.pth"
        
        self.model = EmoTalk(self.args)
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.args["device"])),
            strict=False,
        )
        self.model = self.model.to(self.args["device"])
        self.model.eval()

        self.level = torch.tensor([1]).to(self.args["device"])
        self.person = torch.tensor([0]).to(self.args["device"])

    @torch.no_grad()
    def __call__(self, audio_path: str):
        speech_array, sampling_rate = librosa.load(os.path.join(audio_path), sr=16000)
        audio = torch.FloatTensor(speech_array).unsqueeze(0).to(self.args["device"])
        prediction = self.model.predict(audio, self.level, self.person)
        prediction = prediction.squeeze().detach().cpu().numpy()
        return prediction
    
    
    

