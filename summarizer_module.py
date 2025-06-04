
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch
from symspellpy import SymSpell, Verbosity

# Load mT5 fine-tuned on summarization
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

#ensuring best option as it is a resource intensive task and we dont need delay in outputs
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
model = model.to(device)

# SymSpell setup 
sym_spell = SymSpell(max_dictionary_edit_distance=5, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

def symspell_correction(text):
    words = text.strip().split()
    corrected = []
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)
    return ' '.join(corrected)

# Summarization : seprate pipeline for english and hindi texts which is not widely supported
def summarize_text(text, lang="en", max_length=80):
    if not text.strip():
        return "No text provided."

    # Apply SymSpell spell correction only for English input
    if lang == "en":
        text = symspell_correction(text)

    inputs = tokenizer.encode(text.strip(), return_tensors="pt", max_length=512, truncation=True).to(device)

    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=15,
        num_beams=4,
        repetition_penalty=1.5,
        length_penalty=1.0,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    #return summary 
    return summary