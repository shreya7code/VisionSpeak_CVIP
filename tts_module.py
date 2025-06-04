#importing necessary libraries and modules 
import tempfile
import os
from gtts import gTTS
from TTS.api import TTS
from summarizer_module import summarize_text
from lang_detect_module import detect_language
from transformers import pipeline
import torch

# NLLB multilingual translator
device = 0 if torch.cuda.is_available() else (0 if torch.backends.mps.is_available() else -1)
translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", device=device)

# Function to convert text to speech in specified language
def text_to_speech(text, lang="en", summarize=True):
    if not text.strip():
        return None

    if summarize:
        text = summarize_text(text, lang=lang)
    
    # dual pipeline for the different languages can be extended further for unsupported languages 
    if lang == "en":
        return _tts_english_coqui(text)
    elif lang == "hi":
        return _tts_hindi_gtts(text)
    else:
        raise ValueError(f"Unsupported language: {lang}. Only 'en' and 'hi' are supported.")


def _tts_english_coqui(text):
    # Use Coqui TTS for English
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        tts.tts_to_file(text=text, file_path=f.name)
        return f.name


# different pipeline for unsupported language hindi 
def _tts_hindi_gtts(text):
    try:
        # Translating from Hindi to English using NLLB
        translated = translator(text, src_lang="hin_Deva", tgt_lang="eng_Latn")
        english_text = translated[0]['translation_text']

        # Use English Coqui TTS
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            tts.tts_to_file(text=english_text, file_path=f.name)
            return f.name

    except Exception as e:
        print("Hindi-to-English TTS failed:", e)
        return None