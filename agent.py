import time
import os
from pathlib import Path

from face_module.MICA.configs.config import get_cfg_defaults
from face_module.MICA.MICA import deterministic
from utils.config import config
from utils.savetowav import save_wav
from modules.img2mesh import Image2Mesh
from modules.mesh2blendshape import Mesh2Blendshape
from modules.mesh2talk import Mesh2Talk
from modules.voice2voice import Voice2Voice 

if __name__ == "__main__":
    ### Utils ###
    args = config()   
    os.makedirs(args.output_path, exist_ok=True)
    filename = args.input_audio_path.split('/')[-1].split('.')[0]
    device = args.device
    print(f'torch device set to {device}')
    start_time = time.time()
    
    
    Path(args.o).mkdir(exist_ok=True, parents=True)
    cfg = get_cfg_defaults()
    deterministic(42)
    
    #######################
    
    img2mesh = Image2Mesh(cfg, args)
    mesh2talk  = Mesh2Talk(args)
    voice2voice = Voice2Voice(args)
    
    
    img2mesh()
    result_audio, sampling_rate = voice2voice(args.input_audio_path)
    mesh2talk(speech_array=result_audio, file_name=filename, result_path=args.result_path)
    
    save_wav(result_audio, sampling_rate, filename, args.output_path)