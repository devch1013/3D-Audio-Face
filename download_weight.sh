# export PATH="/usr/local/cuda-12.1/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
gdown 1eXHmBJdYxcvOMkkAnP887ABU0ar-hQKp -O ./face_module/MICA/data/FLAME2020.zip #FLAME2020.zip
unzip ./face_module/MICA/data/FLAME2020.zip -d ./face_module/MICA/data/FLAME2020
gdown 1GQ6VlVClmzuFcNEpgwk87uc_4HHTJqPy -O ./face_module/MICA/data/pretrained/mica.tar #mica.tar
gdown 1BHR4dlltm-G9vaHKXUvV1-koP4nDYPO4 -O ./face_module/LDT/data/fac3Xmodel_head_tri.zip
unzip ./face_module/LDT/data/fac3Xmodel_head_tri.zip -d ./face_module/LDT/data/fac3Xmodel_head_tri
gdown 1KQZ-WGI9VDFLqgNXvJQosKVCbjTaCPqK -O ./face_module/EmoTalk_release/pretrain_model/EmoTalk.pth
gdown 1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT -O ./talk_module/vits/pretrained_ljs.pth
gdown 11aHOlhnxzjpdWDpsz1vFDCzbeEfoIxru -O ./talk_module/vits/pretrained_vctk.pth

# rm ./face_module/MICA/data/FLAME2020.zip
# rm ./face_module/LDT/data/fac3Xmodel_head_tri.zip
