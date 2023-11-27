import whisper
import librosa
from loguru import logger

from talk_module.LLM.chat import ChatGPTConversation
from talk_module.vits.TTS import TextToSpeech, load_TTS


class Voice2Voice:
    def __init__(self, args):
        self.args = args
        self.device = args["general"]["device"]
        logger.info("ㄴ Loading STT model...")
        self.stt_model = whisper.load_model("base").to(self.device)
        self.stt_model.eval()
        logger.info("ㄴ Loading LLM model...")
        self.conversation_agent = ChatGPTConversation()
        logger.info("ㄴ Loading TTS Model...")
        self.tts_model, self.hps = load_TTS(
            config_dir=args["tts_param"]["config_dir"],
            ckpt_dir=args["tts_param"]["tts_ckpt"],
            device=self.device,
        )

    def __call__(self, input_audio_path: str):

        result = self.stt_model.transcribe(input_audio_path)  # m4a, wav 등등 다 가능
        answer = self.conversation_agent(
            "the question is " + result["text"] + ", answer briefly to this question"
        )
        print("answer: ", answer)
        audio, rate = TextToSpeech(self.tts_model, answer, self.hps)

        return (
            librosa.resample(audio, orig_sr=rate[0], target_sr=16000),
            self.hps.data.sampling_rate,
        )
