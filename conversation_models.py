import whisper
from LLM.chat import ChatGPTConversation
import torch
# from tortoise.api import TextToSpeech, MODELS_DIR
from vits.TTS import TextToSpeech, load_TTS
import os
from scipy.io.wavfile import write

# sys.path.append(f"{base_dir}/vits")
base_dir = "/home/ubuntu/3d_temp"
class VoiceAgent:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'torch device set to {device}')

        ###load models###
        print("Loading whisper model...")
        self.stt_model = whisper.load_model("base").to(device)

        print("Loading LLM model...")
        self.conversation_agent = ChatGPTConversation()

        print("Loading TTS Model...")
        self.tts_model, self.hps = load_TTS(config_dir=f"{base_dir}/vits/configs/ljs_base.json",
                            ckpt_dir=f"{base_dir}/vits/pretrained_ljs.pth",
                            device=device)

    def __call__(self, wav_path, output_path = f"{base_dir}/results"):
        print("running Speech to text...")
        result = self.stt_model.transcribe(wav_path)

        print("result of Speech to text...")
        print(result["text"])

        print("running conversation agent...")
        answer = self.conversation_agent(result["text"] + "tell me more about this topic")
        print(answer)

        # answer = "Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth."
        print("running tts model...")
        audio, rate = TextToSpeech(self.tts_model, answer, self.hps)
        wav_filename = wav_path.split('/')[-1].split('.')[0]
        filename = wav_filename + '.wav'
        output_filepath = os.path.join(output_path, filename)
        write(data=audio, rate=self.hps.data.sampling_rate, filename=output_filepath)
        print(f'audio saved to {output_path}')
        return output_filepath