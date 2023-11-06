import time
import argparse
import os

from MICA.configs.config import get_cfg_defaults
from MICA.MICA import MICA, deterministic, process, to_batch, process_single
from pathlib import Path
from face_models import EmotalkFace, Img2Face
from conversation_models import VoiceAgent
import numpy as np
# sys.path.append(f"{base_dir}/vits")
base_dir = "/home/ubuntu/3d_temp"
def config():
    parser = argparse.ArgumentParser()
    #General Parameters
    parser.add_argument("--device", type=str, default="cuda", help='device')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)

    # STT Parameters
    parser.add_argument('--input_audio_path', default=f"{base_dir}/myquestion2.m4a", type=str, help='path of the input wav file')

    #TTS Parameters
    parser.add_argument('--output_path', default = './results', type=str, help='path of the output wav file')
    parser.add_argument('--config_dir', type=str, default = 'vits/configs/ljs_base.json', help='path of the TTS config json file')
    parser.add_argument('--tts_ckpt', type=str, default = 'vits/pretrained_ljs.pth', help='path of the TTS ckpt')
    
    #TTF Parameters
    parser.add_argument("--model_path", type=str, default="EmoTalk_release/pretrain_model/EmoTalk.pth",
                        help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="./result_emotalk", help='path of the result')
    parser.add_argument("--max_seq_len", type=int, default=5000, help='max sequence length')
    parser.add_argument("--post_processing", type=bool, default=True, help='whether to use post processing')
    parser.add_argument("--blender_path", type=str, default="EmoTalk_release/blender", help='path of blender') #Until Here
    parser.add_argument("--bs_dim", type=int, default=52, help='number of blendshapes:52')  #Emotalk Args
    parser.add_argument("--feature_dim", type=int, default=832, help='number of feature dim')
    parser.add_argument("--period", type=int, default=30, help='number of period')
    
    #MICA parameters
    
    parser.add_argument('-path', default=f'{base_dir}/rdj.webp', type=str, help='Input with images') 
    parser.add_argument('-i', default='input', type=str, help='Input folder with images')
    parser.add_argument('-o', default='output_MICA', type=str, help='Output folder')
    parser.add_argument('-a', default='MICA/demo/arcface', type=str, help='Processed images for MICA input')
    parser.add_argument('-m', default='MICA/data/pretrained/mica.tar', type=str, help='Pretrained model path')
    
    #LDT parameters
    parser.add_argument('--ldt_path', default=f"{base_dir}/BlendShapeMaker/Inputs/MICA.ply", type=str, help='path for 3D Avatar')
    args = parser.parse_args()
    return args


class Audio2Face:
    def __init__(self):
        self.face_model = EmotalkFace()
        self.voice_agent = VoiceAgent()
        
        
    def __call__(self, audio_path):
        now = time()
        gpt_audio_path = self.voice_agent(audio_path)
        result = self.face_model(gpt_audio_path)
        np.save(os.path.join(f"{base_dir}/results", "{}.npy".format(audio_path.split("/")[-2])), result)
        print(result)
        print("time: ", time() - now)
        return result
    

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
    img2face = Img2Face(cfg, args)
    img2face()
    
    audio2face = Audio2Face()
    audio2face("/home/ubuntu/3d_temp/myquestion2.m4a")
    
    
    
    
    ###########################

