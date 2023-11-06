import whisper
from conversation.LLM.chat import ChatGPTConversation
import torch
import librosa
import time
import os
os.chdir("/home/ubuntu/3d_temp")

from face_module.EmoTalk_release.demo import load_STF, render_video
from face_module.EmoTalk_release.demo import test as SpeechToFace

# from tortoise.api import TextToSpeech, MODELS_DIR
from conversation.vits.TTS import TextToSpeech, load_TTS

import argparse
import os
import sys
from scipy.io.wavfile import write

from face_module.MICA.configs.config import get_cfg_defaults
from face_module.MICA.MICA import MICA, deterministic, process, to_batch, process_single

from pathlib import Path

import torch
import trimesh
from loguru import logger

from face_module.LDT.gui2cli import LDT

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
    
    parser.add_argument('-path', default=f'{base_dir}/input_images/venedict.jpg', type=str, help='Input with images') 
    parser.add_argument('-i', default='input', type=str, help='Input folder with images')
    parser.add_argument('-o', default='output_MICA', type=str, help='Output folder')
    parser.add_argument('-a', default='MICA/demo/arcface', type=str, help='Processed images for MICA input')
    parser.add_argument('-m', default='MICA/data/pretrained/mica.tar', type=str, help='Pretrained model path')
    
    #LDT parameters
    parser.add_argument('--ldt_path', default=f"{base_dir}/BlendShapeMaker/Inputs/MICA.ply", type=str, help='path for 3D Avatar')
    args = parser.parse_args()
    return args

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
    print("Conver image to face meshes...")
    print("Loading MICA...")
    mica, faces, app = MICA(cfg, args)
    lap_time_0 = time.time()
    print(f"\033[1;3;31mLoading MICA Took... \n\t{lap_time_0 - start_time}s\033[0m")
    
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

        vtx_lmk_idxs = face_tensors[lmk_faces][0][17:,0].tolist() #68개의 Facial Landmark 중 사용하는 51개, 그리고 그 중 Landmark Face 의 3개 Vertex 중 0번째 선택
        data_str = ' '.join(map(str, vtx_lmk_idxs))

        with open(f'{base_dir}/BlendShapeMaker/data/landmarks/landmarks_MICA.txt', 'w') as f:
            f.write(data_str)
        
        mesh = meshes[0]
        # landmark_51 = lmk[0, 17:]
        # landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

        # dst = Path(args.o, name)
        # dst.mkdir(parents=True, exist_ok=True)
        
        
        trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{base_dir}/BlendShapeMaker/Inputs/MICA.ply', file_type='ply', encoding='ascii')  # save in millimeters We only need ply... Save directly to LDT
        
        
        # trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
        # np.save(f'{dst}/identity', code[0].cpu().numpy())   ##### Use if we use flame, or tracker
        # np.save(f'{dst}/kpt7', landmark_7.cpu().numpy() * 1000.0)       ######Landmrak..
        # np.save(f'{dst}/kpt68', lmk.cpu().numpy() * 1000.0)           #####Landmark...

        lap_time_mica = time.time()
        print(f"\033[1;3;31mRunning MICA Took... \n\t{lap_time_mica - lap_time_0}s\033[0m")
        
        print(args.ldt_path)
        LDT(args.ldt_path)
        lap_time_ldt = time.time()
        print(f"\033[1;3;31mRunning LDT Took... \n\t{lap_time_ldt - lap_time_mica}s\033[0m")
    
    ###########################

    ### Load Models  ###
    print("Loading whisper model...")
    stt_model = whisper.load_model("base").to(device)

    print("Loading LLM model...")
    conversation_agent = ChatGPTConversation()

    print("Loading TTS Model...")
    tts_model, hps = load_TTS(config_dir=args.config_dir,
                         ckpt_dir=args.tts_ckpt,
                         device=device)
    
    print("Loading STF Model...")
    stf_model = load_STF(args)
    lap_time_1 = time.time()
    # print(f"\033[1;3;31mLoading Models Took... \n\t{lap_time_1 - lap_time_ldt}s\033[0m")
    
    
    
    ###run models
    print("\033[1;3;33mRunning STT Model...\033[0m")
    
    result = stt_model.transcribe(args.input_audio_path) #m4a, wav 등등 다 가능
    lap_time_2 = time.time()

    print("\033[1;3;33mRunning Conversation Agent...\033[0m")
    answer = conversation_agent(result["text"] + " Tell me more about this topic")
    lap_time_3 = time.time()
    print(f"\033[1;3;32m{answer}\033[0m")
    

    # answer = "Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth."
    print("\033[1;3;33mRunning TTS Model...\033[0m")
    audio, rate = TextToSpeech(tts_model, answer, hps)
    lap_time_4 = time.time()
    
    print("\033[1;3;33mRunning STF Model...\033[0m")

    # #Reload from saved Audio
    # speech_array, _ = librosa.load(save_path, sr=16000)

    speech_array = librosa.resample(audio, orig_sr=rate[0], target_sr=16000)
    SpeechToFace(args, stf_model, speech_array, file_name=filename) #Result path Numpy로 저장 -> 여기서부턴 Unity랑 통신?
    lap_time_5 = time.time()
    
    print("\033[1;3;33mRendering Faces...\033[0m") #

    print(f"\033[1;3;31mRunning Models Took...\n\tSTT : {lap_time_2 - lap_time_1}s\n\tLLM : {lap_time_3 - lap_time_2}s\n\tTTS : {lap_time_4 - lap_time_3}s\n\tSTF : {lap_time_5 - lap_time_4}s\033[0m")

    filename = filename + '.wav'
    save_path = os.path.join(args.output_path, filename)
    write(data=audio, rate=hps.data.sampling_rate, filename=save_path)
    print(f"\033[1;3;33mAudio Saved to : {save_path}\033[0m")
    print(f"\033[1;3;31mTotal Elapsed Time... \n\t{time.time() - start_time}s\033[0m")
