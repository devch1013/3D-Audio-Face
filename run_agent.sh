#!/bin/bash
# echo $CUDA_VISIBLE_DEVICES
export PYTHONVERBOSE=0
# echo $SLURM_NODELIST
# echo $SLURM_NODEID
# ml purge
python agent.py --input_audio_path /home/work/YAI-Summer/junwan/GPT2FACE/VITS/myquestion2.m4a --output_path ./results
#run sh with source
# cp -r ./result_emotalk /root/VITS/EmoTalk_release
# cd EmoTalk_release
# python demo_copy.py --wav_path /root/VITS/results/input.wav
# cd ..