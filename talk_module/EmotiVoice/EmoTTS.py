import re
from frontend_cn import g2p_cn, re_digits, tn_chinese
from frontend_en import ROOT_DIR, read_lexicon, G2p, get_eng_phoneme
import sys
from os.path import isfile

from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder
from transformers import AutoTokenizer
import os, sys, warnings, torch, glob, argparse
import numpy as np
from models.hifigan.get_vocoder import MAX_WAV_VALUE
import soundfile as sf
from yacs import config as CONFIG
from tqdm import tqdm

class TextToSpeech:
    def __init__(self):
        sys.path.append(os.path.dirname(os.path.abspath("__file__")) + "/" + "config/joint")
        from config import Config
        self.config = Config
        self.re_english_word = re.compile('([^\u4e00-\u9fa5]+|[ \u3002\uff0c\uff1f\uff01\uff1b\uff1a\u201c\u201d\u2018\u2019\u300a\u300b\u3008\u3009\u3010\u3011\u300e\u300f\u2014\u2026\u3001\uff08\uff09\u4e00-\u9fa5]+)', re.I)

    def g2p_cn_en(self, text, g2p, lexicon):
        # Our policy dictates that if the text contains Chinese, digits are to be converted into Chinese.
        text=tn_chinese(text)
        parts = self.re_english_word.split(text)
        parts=list(filter(None, parts))
        tts_text = ["<sos/eos>"]
        chartype = ''
        text_contains_chinese = self.contains_chinese(text)
        for part in parts:
            if part == ' ' or part == '': continue
            if re_digits.match(part) and (text_contains_chinese or chartype == '') or self.contains_chinese(part):
                if chartype == 'en':
                    tts_text.append('eng_cn_sp')
                phoneme = g2p_cn(part).split()[1:-1]
                chartype = 'cn'
            elif self.re_english_word.match(part):
                if chartype == 'cn':
                    if "sp" in tts_text[-1]:
                        ""
                    else:
                        tts_text.append('cn_eng_sp')
                phoneme = get_eng_phoneme(part, g2p, lexicon, False).split()
                if not phoneme :
                    # tts_text.pop()
                    continue
                else:
                    chartype = 'en'
            else:
                continue
            tts_text.extend( phoneme )

        tts_text=" ".join(tts_text).split()
        if "sp" in tts_text[-1]:
            tts_text.pop()
        tts_text.append("<sos/eos>")

        return " ".join(tts_text)
    
    def contains_chinese(self, text):
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        match = re.search(pattern, text)
        return match is not None
    
    def phonemize(self, content):
        lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
        g2p = G2p()
        text = self.g2p_cn_en(content.rstrip(), g2p, lexicon)

        return text

    def get_style_embedding(self, prompt, tokenizer, style_encoder):
        prompt = tokenizer([prompt], return_tensors="pt")
        input_ids = prompt["input_ids"]
        token_type_ids = prompt["token_type_ids"]
        attention_mask = prompt["attention_mask"]

        with torch.no_grad():
            output = style_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        style_embedding = output["pooled_output"].cpu().squeeze().numpy()
        return style_embedding
    
    def convert(self, text, content, prompt, speaker, checkpoint):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        root_path = os.path.join(self.config.output_directory, "prompt_tts_open_source_joint")
        ckpt_path = os.path.join(root_path,  "ckpt")
        files = os.listdir(ckpt_path)
        if prompt is None:
            prompt = content
        for file in files:
            if file != checkpoint:
                continue
                
            checkpoint_path = os.path.join(ckpt_path, file)

            with open(self.config.model_config_path, 'r') as fin:
                conf = CONFIG.load_cfg(fin)
            
        
            conf.n_vocab = self.config.n_symbols
            conf.n_speaker = self.config.speaker_n_labels

            style_encoder = StyleEncoder(self.config)
            model_CKPT = torch.load(self.config.style_encoder_ckpt, map_location="cpu")
            model_ckpt = {}
            for key, value in model_CKPT['model'].items():
                new_key = key[7:]
                model_ckpt[new_key] = value
            style_encoder.load_state_dict(model_ckpt, strict=False)

            generator = JETSGenerator(conf).to(device)

            model_CKPT = torch.load(checkpoint_path, map_location=device)
            generator.load_state_dict(model_CKPT['generator'])
            generator.eval()


            with open(self.config.token_list_path, 'r') as f:
                token2id = {t.strip():idx for idx, t, in enumerate(f.readlines())}

            with open(self.config.speaker2id_path, encoding='utf-8') as f:
                speaker2id = {t.strip():idx for idx, t in enumerate(f.readlines())}


            tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)
    
            # if os.path.exists(root_path + "/test_audio/audio/" +f"{file}/"):
            #     r = glob.glob(root_path + "/test_audio/audio/" +f"{file}/*")
            #     for j in r:
            #         os.remove(j)

            style_embedding = self.get_style_embedding(prompt, tokenizer, style_encoder)
            content_embedding = self.get_style_embedding(content, tokenizer, style_encoder)
            if speaker not in speaker2id:
                continue
            speaker = speaker2id[speaker]
            text_int = [token2id[ph] for ph in text.split()]
            sequence = torch.from_numpy(np.array(text_int)).to(device).long().unsqueeze(0)
            sequence_len = torch.from_numpy(np.array([len(text_int)])).to(device)
            style_embedding = torch.from_numpy(style_embedding).to(device).unsqueeze(0)
            content_embedding = torch.from_numpy(content_embedding).to(device).unsqueeze(0)
            speaker = torch.from_numpy(np.array([speaker])).to(device)
            with torch.no_grad():

                infer_output = generator(
                        inputs_ling=sequence,
                        inputs_style_embedding=style_embedding,
                        input_lengths=sequence_len,
                        inputs_content_embedding=content_embedding,
                        inputs_speaker=speaker,
                        alpha=1.0
                    )
                audio = infer_output["wav_predictions"].squeeze()* MAX_WAV_VALUE
                audio = audio.cpu().numpy().astype('int16')

                return audio, self.config.sampling_rate
                # if not os.path.exists(root_path + "/test_audio/audio/" +f"{file}/"):
                #     os.makedirs(root_path + "/test_audio/audio/" +f"{file}/", exist_ok=True)
                # sf.write(file=root_path + "/test_audio/audio/" +f"{file}/{prompt}.wav", data=audio, samplerate=self.config.sampling_rate) #h.sampling_rate
    
    def __call__(self, content, prompt=None, speaker="8051", checkpoint="g_00140000"):
        self.convert(self.phonemize(content), content, prompt, speaker, checkpoint)


if __name__ == "__main__":
    content = "Today is very cold!"
    emotion1 = "惊讶" #普通, 生气, 开心, 惊讶, 悲伤, 厌恶, 恐惧 -> 보통, 화, 즐거움, 놀라움, 슬픔, 혐오, 두려움
    emotion2 = "生气"
    emotion3 = "厌恶"
    emotion4 = "开心"
    TTS = TextToSpeech()
    speaker_id = "1096"
    #input params : input text, (emotional)prompt, speaker id in str, checkpoint name
    TTS(content, emotion1, speaker_id)
    TTS(content, emotion2, speaker_id)
    TTS(content, emotion3, speaker_id)
    TTS(content, emotion4, speaker_id)



