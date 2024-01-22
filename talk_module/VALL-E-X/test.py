from utils.prompt_making import make_prompt

# Number 1 ~ 8 For Emotions 
# 1.Neutral 2.Calm 3.Happy 4.Sad 5.Angry 6.Fearful 7.Disgust 8.Surprised
# This can be expanded as user input voice
emotion_dict = {'neutral':1, 'calm':2, 'happy':3, 'sad':4, 'angry':5, 'fearful':6, 'disgust':7, 'surprised':8}
emotion = 'fearful'

#Use Whisper for transcription
make_prompt(name=emotion, audio_prompt_path=f"./emotion_refs/01-01-0{emotion_dict[emotion]}-02-02-02-01.wav")

from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# download and load all models
preload_models()

text_prompt = """
Hey, Traveler, Listen to this, This machine has taken my voice, and now it can talk just like me!
"""
audio_array = generate_audio(text_prompt, prompt=emotion)

write_wav(f"{emotion}.wav", SAMPLE_RATE, audio_array)
