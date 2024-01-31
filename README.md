### Codes for audio driven 3D face generation

python version: 3.10.13

1. run install.sh file (Not verified yet, so check files and dependencies)
2. run download_weight.sh file
3. place keys.ini at ./talk_module/LLM/keys.ini
4. change line 438 of modelscope.models.cv.face_reconstruction.utils file

    original : os.path.join(save_dir, save_name + '.jpg'), mesh['texture_map']) 
    modified : os.path.join(save_dir, save_name + '.png'), mesh['texture_map'])

    also change line 449 of the same file 

    original : wf.write('map_Kd {}\n'.format(save_name + '.jpg'))
    modified : wf.write('map_Kd {}\n'.format(save_name + '.png'))

    since texturemap in pymeshlab only supports png files

### install lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
cd talk_module/EmotiVoice
git clone https://www.modelscope.cn/syq163/outputs.git
git clone https://www.modelscope.cn/syq163/WangZeJun.git

백엔드 서버 실행
```
uvicorn unity_api.main:app --host 0.0.0.0
```

keys.ini 파일 형태  
[OPENAI]  
OPENAI_API_KEY=key

TODO : 236, 1824번째줄 어떻게 좀 수정하면 빨라질거같은데 둘다 병렬연산될거같아서

VALLEX 넣으면서 추가된내용 > sh 파일들에 아직 미반영
## TODO : VALLE 가 긴 Output은 처리를 못해서 문장 단위로 끊어서 해야할듯
## 레포에도 Max 22 Second라고 하는데 우리지금 계속 돌리는게 20초가량이라 약간 될때도 있고 안될떄도 있음
옛날에 봤던 Bark 같이 hahaha 같은거 넣을 수 있는데 이런거 프롬프트로 넣어도될듯?
