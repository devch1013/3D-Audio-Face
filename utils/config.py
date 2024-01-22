import argparse

def config():
    parser = argparse.ArgumentParser()
    #General Parameters
    parser.add_argument("--device", type=str, default="cuda", help='device')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)

    # STT Parameters
    parser.add_argument('--input_audio_path', default="data/audio_input/myquestion2.m4a", type=str, help='path of the input wav file')

    #TTS Parameters
    parser.add_argument('--output_path', default = 'data/results', type=str, help='path of the output wav file')
    parser.add_argument('--config_dir', type=str, default = 'talk_module/vits/configs/ljs_base.json', help='path of the TTS config json file')
    parser.add_argument('--tts_ckpt', type=str, default = 'talk_module/vits/pretrained_ljs.pth', help='path of the TTS ckpt')
    
    #TTF Parameters
    parser.add_argument("--model_path", type=str, default="face_module/EmoTalk_release/pretrain_model/EmoTalk.pth",
                        help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="data/result_emotalk", help='path of the result')
    parser.add_argument("--max_seq_len", type=int, default=5000, help='max sequence length')
    parser.add_argument("--post_processing", type=bool, default=True, help='whether to use post processing')
    parser.add_argument("--blender_path", type=str, default="EmoTalk_release/blender", help='path of blender') #Until Here
    parser.add_argument("--bs_dim", type=int, default=52, help='number of blendshapes:52')  #Emotalk Args
    parser.add_argument("--feature_dim", type=int, default=832, help='number of feature dim')
    parser.add_argument("--period", type=int, default=30, help='number of period')
    
    #MICA parameters
    
    parser.add_argument('-path', default='data/input_images/venedict.jpg', type=str, help='Input with images') 
    parser.add_argument('-i', default='input', type=str, help='Input folder with images')
    parser.add_argument('-o', default='output_MICA', type=str, help='Output folder')
    parser.add_argument('-a', default='face_module/MICA/demo/arcface', type=str, help='Processed images for MICA input')
    parser.add_argument('-m', default='face_module/MICA/data/pretrained/mica.tar', type=str, help='Pretrained model path')
    
    #LDT parameters
    parser.add_argument('--ldt_path', default=f"face_module/LDT/Inputs/MICA.ply", type=str, help='path for 3D Avatar')
    args = parser.parse_args()
    return args