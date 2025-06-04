VisionSpeak - OCR + Summarization + Translation + Text-to-Speech (Streamlit App)

*** Ensure the python version to be 3.10.13 *** 

How to Run:
1. Install all dependencies:
   pip install -r requirements.txt

2. Ensure OCR model weights and Coqui/NLLB models are downloaded.

3. Launch the Streamlit app:
   bash run.sh

Files:
- app.py: Streamlit frontend entry point
- ocr_module.py: Handles OCR line segmentation and text extraction
- summarizer_module.py: Performs multilingual summarization
- tts_module.py: Handles translation and text-to-speech
- lang_detect_module.py: Detects language to route through correct pipeline