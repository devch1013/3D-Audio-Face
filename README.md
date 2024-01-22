Codes for audio driven 3D face generation

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

백엔드 서버 실행
```
uvicorn unity_api.main:app --host 0.0.0.0
```

keys.ini 파일 형태  
[OPENAI]  
OPENAI_API_KEY=key

TODO : 236, 1824번째줄 어떻게 좀 수정하면 빨라질거같은데 둘다 병렬연산될거같아서

VALLEX 넣으면서 추가된내용 > sh 파일들에 아직 미반영
1. VALLEX의 TTS.py에 init 보면 절대경로로 된거 하나있고
2. talk_module/VALLEX/utils/g2p/__init__.py 에도 하나있음
3. 나머지 weight 같은거는 처음에 실행하면 알아서 받아짐