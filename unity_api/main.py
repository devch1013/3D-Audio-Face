from fastapi import FastAPI, File, UploadFile, Request
import uuid
import random
import numpy as np
import os
from agent import TalkWithMe
from fastapi.responses import FileResponse

app = FastAPI()
print("app loaded")
model = TalkWithMe(conversation_only=False)
# model = None

bs_name = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp_L",
    "browInnerUp_R",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff_L",
    "cheekPuff_R",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]


@app.post("/upload")
async def upload_photo(image: UploadFile):
# async def upload_photo(request: Request):
    # print(await request.form())
    UPLOAD_DIR = "data/input_images"  # 이미지를 저장할 서버 경로
    image_content = await image.read()
    rand_num = str(random.randint(0, 9999999)).zfill(7)
    image = f"{rand_num}.jpg"
    image_path = os.path.join(UPLOAD_DIR, image)
    with open(image_path, "wb") as fp:
        fp.write(image_content)

    model.make_fbx(image_path=image_path, face_name=rand_num)

    return {"id": rand_num}


@app.post("/conversation")
async def conversation(audio: UploadFile):
# async def upload_photo(request: Request):
    # print(await request.form())
    UPLOAD_DIR = "data/audio_input"  # 이미지를 저장할 서버 경로
    audio_content = await audio.read()
    rand_num = str(random.randint(0, 9999999)).zfill(7)
    audio = f"{rand_num}.wav"
    audio_path = os.path.join(UPLOAD_DIR, audio)
    with open(audio_path, "wb") as fp:
        fp.write(audio_content)

    model.conversation(audio_path=audio_path, conversation_name = rand_num)
    # model.conversation(audio_path="data/audio_input/myquestion2.m4a", conversation_name = rand_num)

    return {"id": rand_num}


@app.get("/fbx/{fbx_id}", response_class=FileResponse)
def get_fbx_file(fbx_id: str):
    fbx_path = f"data/result_fbx/{fbx_id}.fbx"
    # fbx_path = "face_module/EmoTalk_release/anima.blend"
    print("fbx path: ", fbx_path)
    return fbx_path


@app.get("/texture/{texture_id}", response_class=FileResponse)
def get_texture_file(texture_id: str):
    texture_path = f"data/result_fbx/{texture_id}.png"
    # texture_path = "face_module/EmoTalk_release/anima.blend"
    print("texture path: ", texture_path)
    return texture_path


@app.get("/audio/{audio_id}", response_class=FileResponse)
def get_audio_file(audio_id: str):
    audio_path = f"data/results/{audio_id}.wav"
    return audio_path


@app.get("/bsweight/{bsweight_id}")
def get_bsweight_file(bsweight_id: str):
    bsweight_path = f"data/result_emotalk/{bsweight_id}.npy"
    bsweight_list = np.load(bsweight_path).tolist()
    print(len(bsweight_list[0]))
    for i in range(len(bsweight_list)):
        bsweight_list[i].insert(6, bsweight_list[i][6])
        bsweight_list[i].insert(2, bsweight_list[i][2])
        bsweight_list[i].pop(-1)
    return {"length":len(bsweight_list),"data": bsweight_list}
