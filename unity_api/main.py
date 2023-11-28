from fastapi import FastAPI, File, UploadFile
import uuid
import random
import numpy as np
import os
from agent import TalkWithMe
from fastapi.responses import FileResponse

app = FastAPI()

# model = TalkWithMe()
model = None

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
async def upload_photo(image: UploadFile, audio: UploadFile):
    UPLOAD_DIR = "data/"  # 이미지를 저장할 서버 경로

    image_content = await image.read()
    audio_content = await audio.read()
    rand_num = str(random.randint(0, 9999999)).zfill(7)
    image = f"input_images/{rand_num}.jpg"
    audio = f"audio_input/{rand_num}.m4a"
    image_path = os.path.join(UPLOAD_DIR, image)
    audio_path = os.path.join(UPLOAD_DIR, audio)
    with open(image_path, "wb") as fp:
        fp.write(image_content)
    with open(audio_path, "wb") as fp:
        fp.write(audio_content)

    model(image_path=image_path, input_audio_path=audio_path, face_name=rand_num)

    return {"id": rand_num}


@app.get("/obj/{obj_id}", response_class=FileResponse)
def get_obj_file(obj_id: str):
    obj_path = f"/home/ubuntu/3d_temp/face_module/LDT/Inputs/{obj_id}.ply"
    return obj_path


@app.get("/audio/{audio_id}", response_class=FileResponse)
def get_audio_file(audio_id: str):
    audio_path = f"/home/ubuntu/3d_temp/data/results/{audio_id}.wav"
    return audio_path


@app.get("/bsweight/{bsweight_id}")
def get_bsweight_file(bsweight_id: str):
    bsweight_path = f"/home/ubuntu/3d_temp/data/result_emotalk/{bsweight_id}.npy"
    bsweight_list = np.load(bsweight_path).tolist()
    # print(len(bsweight_list))
    # bs_dict = get_bs_dict(bsweight_list)
    return {"length":len(bsweight_list),"data": bsweight_list}


def get_bs_dict(bs_list):
    bs_dict = dict()
    for bs in bs_name:
        bs_dict[bs] = list()
        
    for i in range(len(bs_list)):
        for j in range(len(bs_name)):
            if j > 2 and j <= 6:
                new_idx = j-1
            elif j > 6:
                new_idx = j-2
            else:
                new_idx = j
            bs_dict[bs_name[j]].append(bs_list[i][new_idx])
    return bs_dict