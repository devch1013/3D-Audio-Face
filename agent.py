import time
import os
from pathlib import Path

from loguru import logger

from face_module.MICA.configs.config import get_cfg_defaults
from face_module.MICA.MICA import deterministic
from utils.config import config
from utils.savetowav import save_wav
from modules.img2mesh import Image2Mesh
from modules.mesh2blendshape import Mesh2Blendshape
from modules.mesh2talk import Mesh2Talk
from modules.voice2voice import Voice2Voice 

class TalkWithMe:
    def __init__(self):
        ### Utils ###
        self.args = config()   
        os.makedirs(self.args.output_path, exist_ok=True)
        self.filename = self.args.input_audio_path.split('/')[-1].split('.')[0]
        device = self.args.device
        logger.info(f'torch device set to {device}')
        self.start_time = time.time()
        
        Path(self.args.o).mkdir(exist_ok=True, parents=True)
        cfg = get_cfg_defaults()
        deterministic(42)
        
        #######################
        start = time.time()
        logger.info("Load Image2Mesh Model")
        cp = time.time()
        self.img2mesh = Image2Mesh(cfg, self.args)
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
    
    def __call__(self, image_path, input_audio_path):
        start = time.time()
        
        logger.info("Make Mesh from Image")
        cp = time.time()
        self.img2mesh(image_path)
        print(f"\033[1;3;31mRunning Image2Mesh Took... \n\t{time.time() - cp}s\033[0m")
        
        logger.info("Make Blendshapes")
        cp = time.time()
        Mesh2Blendshape(self.args.ldt_path)
        print(f"\033[1;3;31mRunning Mesh2Blendshape Took... \n\t{time.time() - cp}s\033[0m")
        
        logger.info("Conversation Part Started")
        cp = time.time()
        result_audio, sampling_rate = self.voice2voice(input_audio_path)
        print(f"\033[1;3;31mRunning Voice2Voice Took... \n\t{time.time() - cp}s\033[0m")
        
        logger.info("Make Talking Face with Audio")
        cp = time.time()
        self.mesh2talk(speech_array=result_audio, file_name=self.filename, result_path=self.args.result_path)
        print(f"\033[1;3;31mRunning Mesh2Talk Took... \n\t{time.time() - cp}s\033[0m")
        save_wav(result_audio, sampling_rate, self.filename, self.args.output_path)
        
        logger.info("Finish Process")
        print(f"\033[1;3;31mRunning Process Took... \n\t{time.time() - start}s\033[0m")
        
        
        
if __name__ == "__main__":
    main_model = TalkWithMe()
    image_path = "data/input_images/venedict.jpg"
    input_audio_path = "data/audio_input/myquestion2.m4a"
    main_model(image_path, input_audio_path)