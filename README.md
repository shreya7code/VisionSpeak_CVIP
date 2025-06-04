# VisionSpeak
AI-Powered Multi-Language OCR with Instant Summarization &amp; Speech

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Project Information: Web Application Featuring Optical Character Recognition, Language Detection, Summarization and Text-to-Speech Functionality 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Need to Work: 
1) We need to detect various types of languages (Unsupported & Supported) with OCR.  ( Major Work)
2) Gather Dataset 
3) Pre-process Dataset 
4) Prepare the dataset for training 

Add to report: scalable ML pipeline for real-time document processing in enterprises for accessibility and workflow optimization.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

For Unsupported Language: Use fasttext (unpopular language detection) --> NLLB (unpopular lang translation to eng ) ---> T5 / ( Summerization in English and using TTS.api voice synthesis in English) ----> NLLB ( Convert to unpopular lang).

Popular language / Supported by t5: We will directly call the required models which is best suited for the specific language by passing it as a parameter.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Planned OCR Network Architecture (Final Model)

✅ Final Model: ResNet-100 + CRNN + BiLSTM + CTC
	•	Network Architecture:
	•	ResNet-100 (Deep CNN) as a feature extractor (captures text patterns and structures).
	•	CRNN (Convolutional Recurrent Neural Network) for sequence modeling.
	•	BiLSTM (Bidirectional Long Short-Term Memory) to handle context in handwritten & multi-line text.
	•	CTC (Connectionist Temporal Classification) Loss for end-to-end text decoding.
Why This Model?
	•	Strong performance on both printed and handwritten text.
	•	Handles noisy, distorted, and irregular layouts better than traditional OCR.
	•	Comparison with ResNet-50 + CRNN evaluates the impact of depth on OCR accuracy.

1️⃣ Baseline Model 1: ResNet-50 + CRNN + BiLSTM + CTC
	•	Network Architecture:
	•	ResNet-50 (CNN) as a shallower feature extractor.
	•	CRNN + BiLSTM + CTC for text sequence modeling (same as the final model).
	•	Purpose of Comparison:
	•	Tests how network depth impacts OCR performance.
	•	Helps determine if ResNet-100’s additional complexity is justified.
	•	Measures trade-offs in accuracy vs. computational cost.

2️⃣ Baseline Model 2: TrOCR (Transformer-Based OCR)
	•	Network Architecture:
	•	ViT (Vision Transformer) Encoder replaces CNN for feature extraction.
	•	Transformer Decoder for text sequence generation (similar to GPT).
	•	Purpose of Comparison:
	•	Tests whether CNN + LSTM (ResNet-100 + CRNN) or Transformers generalize better for OCR tasks.
	•	Evaluates state-of-the-art Transformer-based OCR.
	•	Stronger on printed text but requires higher computational power.

