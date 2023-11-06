import whisper
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
model = whisper.load_model("base").to(device)
result = model.transcribe("/home/whwjdqls99/nonverbal_cues/Dataset/noisey/laughing/e1rZP6CzMnY.wav")
print(result["text"])