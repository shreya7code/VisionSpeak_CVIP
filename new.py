#!/bin/bash

# Step 1: Set Python 3.11 path (Homebrew)
PY311=/opt/homebrew/opt/python@3.11/bin/python3
PIP311=/opt/homebrew/opt/python@3.11/bin/pip3

# Step 2: Download sentencepiece wheel for Python 3.11 macOS arm64
echo "📥 Downloading sentencepiece wheel..."
curl -L -o sentencepiece.whl https://files.pythonhosted.org/packages/1e/69/53b1b3cb23686bc78e967c7d51d4174fdb7bd2dbe388b06600f7b2b85800/sentencepiece-0.1.99-cp311-cp311-macosx_11_0_arm64.whl

# Step 3: Install wheel with pip3 + override Homebrew protection
echo "📦 Installing sentencepiece..."
$PIP311 install ./sentencepiece.whl --break-system-packages

# Step 4: Optional - install your other dependencies
echo "📦 Installing project dependencies..."
$PIP311 install -r requirements.txt --break-system-packages

# Step 5: Run the app using Python 3.11
echo "🚀 Launching app with Python 3.11..."
$PY311 -m streamlit run app.py