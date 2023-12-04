import time
import os
from pathlib import Path

from loguru import logger
import yaml
import torch
import open3d as o3d

from face_module.MICA.configs.config import get_cfg_defaults
from face_module.MICA.MICA import deterministic
from face_module.HRN import HRN
from utils.config import config
from utils.savetowav import save_wav
from modules.img2mesh import Image2Mesh
from modules.mesh2blendshape import Mesh2Blendshape
from modules.mesh2talk import Mesh2Talk
from modules.voice2voice import Voice2Voice 


class TalkWithMe:
    def __init__(self, create_new=True):
        ### Utils ###
        with open('utils/configs.yml', 'r') as file:
            self.args = yaml.safe_load(file)
        
        
        device = self.args["general"]["device"]
        logger.info(f'torch device set to {device}')
        self.start_time = time.time()
        self.create_new = create_new
    
        deterministic(42)
        
        start = time.time()
        if self.create_new:
            self.create_new = True
            logger.info("Load Image2Mesh Model")
            cp = time.time()

            self.img2mesh = HRN(output_dir='./hrn_output')
            print(f"\033[1;3;31mLoading Img2Mesh Took... \n\t{time.time() - cp}s\033[0m")
        
            logger.info("Load Mesh2Talk Model")
        cp = time.time()

        self.mesh2talk  = Mesh2Talk(self.args)
        print(f"\033[1;3;31mLoading Mesh2Talk Took... \n\t{time.time() - cp}s\033[0m")
        
        logger.info("Load Voice2Voice Model")
        cp = time.time()
        self.voice2voice = Voice2Voice(self.args)
        print(f"\033[1;3;31mLoading Voice2Voice Took... \n\t{time.time() - cp}s\033[0m")
        logger.info("Model Load Finished")
        print(f"\033[1;3;31mLoading Models Took... \n\t{time.time() - start}s\033[0m")
    
    def __call__(self, image_path, input_audio_path, face_name):
        filename = input_audio_path.split('/')[-1].split('.')[0]
        start = time.time()
        
        with torch.no_grad():
            if self.create_new:
                logger.info("Make Mesh from Image")
                cp = time.time()
                self.img2mesh(face_name, image_path)

                print(f"\033[1;3;31mRunning Image2Mesh Took... \n\t{time.time() - cp}s\033[0m")
                
                logger.info("Make Blendshapes")
                cp = time.time()
                Mesh2Blendshape(os.path.join(self.args["mica_param"]["output_path"], f"{face_name}.ply"))
                print(f"\033[1;3;31mRunning Mesh2Blendshape Took... \n\t{time.time() - cp}s\033[0m")

                logger.info("Adding Textures Started")
                cp = time.time()
                self.img2mesh.make_textures()
                print(f"\033[1;3;31mConverting to obj file with textures Took... \n\t{time.time() - cp}s\033[0m")
        
            logger.info("Conversation Part Started")
            cp = time.time()
            result_audio, sampling_rate = self.voice2voice(input_audio_path)
            print(f"\033[1;3;31mRunning Voice2Voice Took... \n\t{time.time() - cp}s\033[0m")
            print("sampling rate: ",sampling_rate)

            logger.info("Make Talking Face with Audio")
            cp = time.time()
            self.mesh2talk(speech_array=result_audio, file_name=filename, result_path=self.args["ttf_param"]["result_path"])
            print(f"\033[1;3;31mRunning Mesh2Talk Took... \n\t{time.time() - cp}s\033[0m")
            print(self.args["tts_param"]["output_path"],":",filename)
            save_wav(result_audio, sampling_rate, filename, self.args["tts_param"]["output_path"])

            logger.info("Finish Process")
            print(f"\033[1;3;31mRunning Process Took... \n\t{time.time() - start}s\033[0m")
        
        
        
if __name__ == "__main__":
    main_model = TalkWithMe(create_new=False)
    image_path = "data/input_images/murphy.png"
    input_audio_path = "data/audio_input/myquestion2.m4a"
    face_name = "murphy"
    main_model(image_path, input_audio_path, face_name)