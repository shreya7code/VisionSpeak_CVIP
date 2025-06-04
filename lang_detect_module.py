import fasttext
import os

# load the pre-trained FastText language detection model
MODEL_PATH = "lid.176.bin"
model = fasttext.load_model(MODEL_PATH)

def detect_language(text):
    if not text.strip():
        return "Language not yet Supported"
    
    # Remove newlines and excessive whitespace 
    cleaned_text = text.replace('\n', ' ').strip()

    predictions = model.predict(cleaned_text)
    lang_code = predictions[0][0].replace('__label__', '')
    return lang_code