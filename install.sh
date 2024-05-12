# export PATH="/usr/local/cuda-12.1/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
pip install -r requirements.txt # Works on CUDA 12.1 & Pytorch 2.1.1
wget https://files.pythonhosted.org/packages/4a/ca/e85e628105c2f80412231436ff0644f7dc96630520f4a7c4eb2b3a6085f2/bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl
pip install bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl && bpy_post_install
# 
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
sudo apt-get install ffmpeg
sudo apt-get install python3-opencv
sudo apt-get install espeak
# rm ./face_module/MICA/data/FLAME2020.zip
# rm ./face_module/LDT/data/fac3Xmodel_head_tri.zip
