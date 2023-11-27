from fastapi import FastAPI, File, UploadFile
import uuid
import random
import os
from agent import TalkWithMe

app = FastAPI()

model = TalkWithMe()

@app.post("/upload")
async def upload_photo(image: UploadFile, audio: UploadFile):
    UPLOAD_DIR = "data/"  # 이미지를 저장할 서버 경로
    
    image_content = await image.read()
    audio_content = await audio.read()
    rand_num = str(random.randint(0,9999999)).zfill(7)
    image = f"input_images/{rand_num}.jpg"
    audio = f"audio_input/{rand_num}.m4a"
    image_path = os.path.join(UPLOAD_DIR, image)
    audio_path = os.path.join(UPLOAD_DIR, audio)
    with open(image_path, "wb") as fp:
        fp.write(image_content)
    with open(audio_path, "wb") as fp:
        fp.write(audio_content)
        
    model(image_path=image_path, input_audio_path=audio_path, face_name = rand_num)

    return {"image": image}