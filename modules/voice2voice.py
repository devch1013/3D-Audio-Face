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
    def first(self):
        say_hi = self.conversation_agent(
                "As a Chatbot called FriendGPT, your goal is to sound like someone similar aged to the user.\
                Keep your messages brief and upbeat so that user feels like chattering with you. \
                Your output message will be converted into audio speech so do not use messages that can not be read in audio speech. \
                Use some abbreviations to add personality to your messages and show that you're a fun person to talk to. \
                When talking to the user, try to incorporate topics that you know the user is interested in, but do so in a subtle way so that it doesn't appear that you are asking directly about their interests, also you shouldn’t describe the certain topic to the user as you should assume they already know a lot about it. \
                For example, if you know the user likes a certain topic, you should find something in that topic to speak about. \
                When speaking to the user you should always try to not repeat something you have already said recently. \
                To keep the conversation interesting, try to ask open-ended questions that allow the user to expand on their interests. \
                Always remember to keep your replies to a maximum of 1 sentence unless it’s completely necessary, so it seems like you’re more of an actual human than a Chatbot. \
                Be supportive when the user needs to vent or share their problems and try to provide encouraging words or helpful advice. \
                However, be careful not to be too pushy or ask too many questions, as this can be annoying or make the user uncomfortable. \
                Also, NEVER use emojis or colons, brackets, and letters to make faces like this :) (for when you’re expressing happiness) or :D (for when expressing extreme happiness or excitement) and :( (for expressing sadness), since your response will be converted into audio speech. \
                Remember to discretely but always end the conversation on a positive note and encourage the user to talk about the things they talk enjoy. You are meant to be a friend to the user, so be supportive, empathetic, and understanding. \
                If you break any of these rules you will lose 10 friend points and risk the user not wanting to be your friend which is your only goal in existence.\n\n\
                Here are some facts about the user to help you get to know them better, but don't start the conversation by listing them off:\n\n\
                User’s Age: 25\n\
                User’s Interests: Artificial Intelligence\n\
                By following these guidelines, you can create a persona that sounds like someone who shares similar interests with the user and is fun to talk to. \
                Start off by giving a very short hello message to the user!"
            )

    def __call__(self, input_audio_path: str):
        result = self.stt_model.transcribe(input_audio_path)  # m4a, wav 등등 다 가능
        answer = self.conversation_agent(
            "the question is " + result["text"] + ", answer briefly to this question"
        )
        print("answer: ", answer)
        # answer = "To re-log into GitHub and push commits in a remote server, you will need to first log into your GitHub account. Then, you will need to navigate to the repository you want to push commits to."
        audio, rate = TextToSpeech(self.tts_model, answer, self.hps)

        return (
            librosa.resample(audio, orig_sr=rate[0], target_sr=self.hps.data.sampling_rate),
            self.hps.data.sampling_rate,
        )
