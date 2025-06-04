# import necessary libraries and modules 

import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
from PIL import Image
from ocr_module import extract_text_from_image
from lang_detect_module import detect_language
from summarizer_module import summarize_text
from tts_module import text_to_speech

st.set_page_config(page_title="VisionSpeak", layout="wide")

# styling for the front end #
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #1e1e1e;
    padding-top: 1rem;
}
h1 {
    font-size: 3rem;
    text-align: center;
    margin-bottom: 0.2em;
}
h4 {
    text-align: center;
    color: #aaa !important;
    font-weight: 400;
    margin-bottom: 1.5em;
}
section[data-testid="stFileUploader"] {
    border-radius: 10px;
    border: 1px solid #444;
    background-color: #2c2c2c;
    padding: 1.5rem;
}
div[data-testid="stFileUploader"] > label {
    font-weight: 600;
    color: #fff;
}
#custom-toast {
    visibility: hidden;
    min-width: 150px;
    background-color: #28a745 !important;
    color: #fff;
    text-align: center;
    border-radius: 8px;
    padding: 10px;
    position: fixed;
    z-index: 1000;
    left: 50%;
    transform: translateX(-50%);
    bottom: 30px;
    font-size: 15px;
    box-shadow: 0 0 10px rgba(40, 167, 69, 0.7);
}
#custom-toast.show {
    visibility: visible;
    animation: fadein 0.3s, fadeout 0.5s 1.5s;
}
@keyframes fadein { from {bottom: 20px;opacity: 0;} to {bottom: 30px;opacity: 1;} }
@keyframes fadeout { from {bottom: 30px;opacity: 1;} to {bottom: 20px;opacity: 0;} }
</style>
<div id="custom-toast"></div>
""", unsafe_allow_html=True)

#  Header Section #
st.markdown("<h1>VisionSpeak</h1>", unsafe_allow_html=True)
st.markdown("<h4>AI-powered Document Processor: OCR,Language Detector,Summarization, and Text-to-Speech</h4>", unsafe_allow_html=True)
st.markdown("---")

# doxument upload section 
st.subheader("Upload Document")
uploaded_file = st.file_uploader("Upload a scanned document or image", type=["jpg", "jpeg", "png"])

# processing after file upload
if uploaded_file:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img_path = tmp.name
        img.save(img_path)

    with st.spinner("Extracting text..."):
        extracted_text = extract_text_from_image(img_path, model_path="best_model_resnet101_v2.pth")

    with col2:
        st.subheader("Extracted Text")
        st.code(extracted_text, language='markdown')

        components.html(f"""
            <div style="text-align: center;">
              <textarea id="ocrText" style="display:none;">{extracted_text}</textarea>
              <button onclick="
                navigator.clipboard.writeText(document.getElementById('ocrText').value);
                let toast = document.getElementById('custom-toast');
                toast.innerText = 'Copied Extracted Text!';
                toast.className = 'show';
                setTimeout(() => {{ toast.className = toast.className.replace('show', ''); }}, 2000);
              "
              style="background-color:#4CAF50;border:none;color:white;padding:10px 24px;
                     font-size:16px;border-radius:8px;cursor:pointer;margin-top:10px;">
                Copy Extracted Text
              </button>
            </div>
        """, height=100)

    st.markdown("---")

    st.subheader("Language Detection")
    with st.spinner("Detecting language..."):
        lang = detect_language(extracted_text)
    st.write(f"Detected Language: `{lang}`")

    st.markdown("---")

    st.subheader("Summarization")
    with st.spinner("Summarizing..."):
        summary = summarize_text(extracted_text, lang)

    st.code(summary, language='markdown')

    # adding html component for copy button
    components.html(f"""
        <div style="text-align: center;">
          <textarea id="summaryText" style="display:none;">{summary}</textarea>
          <button onclick="
            navigator.clipboard.writeText(document.getElementById('summaryText').value);
            let toast = document.getElementById('custom-toast');
            toast.innerText = 'Copied Summary!';
            toast.className = 'show';
            setTimeout(() => {{ toast.className = toast.className.replace('show', ''); }}, 2000);
          "
          style="background-color:#4CAF50;border:none;color:white;padding:10px 24px;
                 font-size:16px;border-radius:8px;cursor:pointer;margin-top:10px;">
            Copy Summary
          </button>
        </div>
    """, height=100)

    st.markdown("---")

    st.subheader("Text-to-Speech")
    with st.spinner("Generating audio..."):
        # prevent re-summarizing in the TTS step
        audio_path = text_to_speech(summary, lang, summarize=False)

    if os.path.exists(audio_path):
        st.audio(audio_path, format='audio/mp3')
        with open(audio_path, 'rb') as f:
            st.download_button("Download Audio", f, file_name="summary_audio.mp3", mime='audio/mp3')

    os.remove(img_path)

else:
    st.info("Please upload an image to begin processing.")