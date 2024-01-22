from talk_module.VALLEX.vallex_utils.prompt_making import make_prompt
from talk_module.VALLEX.vallex_utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

def init_emotion():
    # 1.Neutral 2.Calm 3.Happy 4.Sad 5.Angry 6.Fearful 7.Disgust 8.Surprised
    # This can be expanded as user input voice
    emotion_dict = {'neutral':1, 'calm':2, 'happy':3, 'sad':4, 'angry':5, 'fearful':6, 'disgust':7, 'surprised':8}
    for emotion in emotion_dict.keys():
        make_prompt(name=emotion, audio_prompt_path=f"./talk_module/VALLEX/emotion_refs/01-01-0{emotion_dict[emotion]}-02-02-02-01.wav")
    preload_models()

def TextToSpeech(emotion, input_text):
    audio_array = generate_audio(input_text, prompt=emotion)
    return audio_array, SAMPLE_RATE