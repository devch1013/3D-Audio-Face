import whisper
import librosa

from talk_module.LLM.chat import ChatGPTConversation
from talk_module.vits.TTS import TextToSpeech, load_TTS


class Voice2Voice:
    def __init__(self, args):
        self.args = args
        device = "cuda"
        self.stt_model = whisper.load_model("base").to(device)
        print("Loading LLM model...")
        self.conversation_agent = ChatGPTConversation()

        print("Loading TTS Model...")
        self.tts_model, self.hps = load_TTS(
            config_dir=args.config_dir, ckpt_dir=args.tts_ckpt, device=device
        )
        
    def __call__(self, input_audio_path:str):
        result = self.stt_model.transcribe(self.args.input_audio_path) #m4a, wav 등등 다 가능
        answer = self.conversation_agent(result["text"] + " Tell me more about this topic")
        audio, rate = TextToSpeech(self.tts_model, answer, self.hps)
        
        return librosa.resample(audio, orig_sr=rate[0], target_sr=16000), self.hps.data.sampling_rate
        
        