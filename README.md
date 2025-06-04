# VisionSpeak

**VisionSpeak** is an AI-powered multimodal pipeline that integrates Optical Character Recognition (OCR), language detection, text summarization, and Text-to-Speech (TTS) into a seamless system. It is built to assist users with visual impairments or language barriers by extracting and vocalizing multilingual handwritten text from documents and forms.

---

## 🌟 Core Features

- 🖋️ **Handwritten Text Recognition** using a fine-tuned `microsoft/trocr-base-handwritten` Transformer model.
- 🌍 **Auto Language Detection** with FastText to support multilingual content.
- 🧠 **Contextual Summarization** using T5 to extract meaningful gist from recognized content.
- 🗣️ **Natural Speech Output** through Mozilla TTS for personalized, clear, and multilingual voice support.
- 🧩 **Modular Pipeline** that connects OCR → Language Detection → Summarization → TTS in real time.

---

## 🏗️ Architecture Overview

- **Vision Encoder**: ViT-based encoder from TrOCR to extract visual features.
- **Text Decoder**: GPT-style decoder to transcribe visual inputs into text.
- **Tokenizer**: Custom character-level tokenizer for handling multilingual handwritten data.
- **Pipeline**:
  1. Image preprocessing (grayscale, padding, normalization)
  2. OCR (TrOCR-based transcription)
  3. Language detection (FastText)
  4. Translation (NLLB if needed)
  5. Summarization (T5)
  6. Text-to-Speech (Mozilla TTS)

---

## 📊 Evaluation Metrics

| Task              | Metric                | Target            |
|-------------------|------------------------|--------------------|
| OCR               | Character Error Rate (CER), Word Error Rate (WER) | CER ≤ 5% |
| Summarization     | ROUGE Score            | ROUGE ≥ 0.8        |
| Speech Generation | Mean Opinion Score (MOS) | MOS ≥ 4.5 / 5     |
| Real-Time Response| Processing Time        | OCR ≤ 1s/page, Summarization ≤ 3s/500 tokens |

---

## ✅ Positive Test Cases

- High-resolution handwritten forms with clean structure
- Monolingual and well-formatted content
- Common languages supported by TTS engine

## ⚠️ Failure Scenarios

- Noisy, low-resolution, or distorted handwriting
- Mixed-language paragraphs with abrupt switches
- Technical documents with domain-specific jargon
- Rare dialects not supported by TTS, leading to robotic pronunciation

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/VisionSpeak.git
   cd VisionSpeak

2. Install Dependecies:
   ```bash
   pip install -r requirements.txt

3. Run the application:
   ```bash
   python app.py


