# importing importent libraries and modules

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models
from symspellpy.symspellpy import SymSpell, Verbosity
import os
import torch
import re
#from ocr_module import CRNN_OCR_ResNet101_v2, OCRTokenizer, segment_lines, preprocess_line_image, symspell_correction

# defining the OCRTokenizer class that will be used for encoding and decoding text
# the same tokenizer was used during the training of the OCR model
class OCRTokenizer:
    def __init__(self, from_config):
        self.char2idx = from_config
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def encode(self, text):
        return [self.char2idx[char] for char in text if char in self.char2idx]

    def decode(self, indices):
        return "".join([self.idx2char.get(idx, '') for idx in indices if idx > 0])

    def __len__(self):
        return len(self.char2idx)

# post correction using SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_loaded = sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
if not dictionary_loaded:
    raise RuntimeError("SymSpell dictionary file not found.")

# the correction function that uses SymSpell to correct the text
def symspell_correction(text):
    words = text.split()
    corrected = []
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)
    return ' '.join(corrected)

# model definition using ResNet101 as the backbone
class CRNN_OCR_ResNet101_v2(nn.Module):
    def __init__(self, num_classes):
        super(CRNN_OCR_ResNet101_v2, self).__init__()
        resnet = models.resnet101(pretrained=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-3])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            input_size=1024,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        features = self.cnn(x)
        features = self.adaptive_pool(features)
        features = features.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(features)
        output = self.fc(rnn_out).permute(1, 0, 2)
        return output

# Necessary Preprocessing of the input image from the frontend
def preprocess_line_image(image_array, size=(2048, 128)):
    resized = cv2.resize(image_array, size)
    normalized = resized.astype(np.float32) / 255.0
    normalized = np.expand_dims(normalized, axis=0)
    return normalized

# applying line segmentation logic to the input image
def segment_lines(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    hist = np.sum(binary, axis=1)
    height, width = binary.shape
    in_line = False
    top = 0
    line_images = []

    for row in range(height):
        if hist[row] > 0 and not in_line:
            in_line = True
            top = row
        elif hist[row] == 0 and in_line:
            in_line = False
            bottom = row
            if bottom - top > 5:
                line_img = gray[top:bottom, :]
                line_images.append(line_img)

    return line_images

# inference function to extract text from the image using the trained OCR model
def extract_text_from_image(image_path, model_path="best_model_resnet101_v2.pth"):
    # Ensuring uniformity in device usage because MPS is not widely supported, e.g., on ARM-based Macs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    checkpoint = torch.load(model_path, map_location=device)

    # Loading the tokenizer using the configuration 
    tokenizer = OCRTokenizer(from_config=checkpoint['tokenizer_config'])

    # Loading the model using the checkpoint
    model = CRNN_OCR_ResNet101_v2(num_classes=checkpoint['vocab_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Segmentation of lines from input image using segment_lines function
    line_images = segment_lines(image_path)
    results = []

    for line in line_images:
        # Preprocessing of each segmented line image
        processed = preprocess_line_image(line)
        tensor = torch.tensor(processed, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            predictions = logits.softmax(2).argmax(2).permute(1, 0)
            decoded = tokenizer.decode(predictions[0].cpu().numpy())

            # Applying SymSpell only for English
            if "<en>" in decoded.lower():
                corrected = symspell_correction(decoded)
            else:
                corrected = decoded

            # cleaning up the text to remove any tags
            cleaned = re.sub(r'<(en|hi)[^>]*>', '', corrected, flags=re.IGNORECASE)
            cleaned = re.sub(r'[<>]+', '', cleaned)
            cleaned = cleaned.strip()
            cleaned = re.sub(r'^(send|this)\b\s*', '', cleaned, flags=re.IGNORECASE) 

            results.append(cleaned)

    return "\n".join(results)
