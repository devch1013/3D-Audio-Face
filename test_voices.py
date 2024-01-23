from talk_module.EmotiVoice.EmoTTS import TextToSpeech as TextToSpeechEmoti
import soundfile as sf
import os
from tqdm import tqdm

tts_model = TextToSpeechEmoti()
answer = "Today is very cold!"
speakers = ['9000','984','985','65','92','102','225','1088','1093']
prompts_cn = ['普通', '生气', '开心', '惊讶', '悲伤', '厌恶', '恐惧']
prompts_en = ['Neutral', 'Angry', 'Happy', 'Surprised', 'Sad', 'Disgusted', 'Fearful']
for speaker in tqdm(speakers):
    os.makedirs(f"./test_voices/{speaker}", exist_ok=True)
    for prompt in prompts_en:
        audio, rate = tts_model(answer, prompt=prompt, speaker=speaker)
        sf.write(f"./test_voices/{speaker}" +f"/{prompt}.wav", data=audio, samplerate=rate)
