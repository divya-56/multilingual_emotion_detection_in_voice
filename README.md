# multilingual_emotion_detection_in_voice

---

# 🧠🎙️ Multilingual Emotion Detection in Voice

This project builds an intelligent system that detects **emotions in speech** and **transcribes multilingual audio** using OpenAI's Whisper model. It supports voice inputs from video/audio files, extracts acoustic features, and classifies emotions using a machine learning model.

---

## 🚀 Features

* 🎧 Transcribes speech from `.mp4` or `.wav` files using OpenAI Whisper.
* 🌍 Automatically detects the **language** of the speech.
* 🎭 Extracts **audio features** (MFCC + pitch) using `librosa`.
* 💡 Classifies the speaker's **emotional state** (*happy*, *sad*, *angry*, *neutral*) using an SVM model.
* 🔁 Converts MP4 video files into audio format for analysis.

---

## 🧰 Requirements

Install all dependencies using:

```bash
pip install openai-whisper librosa scikit-learn moviepy
```

---

## 📂 Project Workflow

### Step 1: Transcribe Speech

* Uses `whisper` to convert speech to text.
* Returns both the transcription and detected language.

### Step 2: Extract Features

* MFCC (Mel-frequency cepstral coefficients)
* Pitch tracking using `librosa.piptrack`
* Combines both features for emotion recognition

### Step 3: Train Emotion Classifier

* Uses simulated data for demonstration (replace with real labeled data for production).
* Uses `SVC` (Support Vector Classifier) to predict emotional categories.

### Step 4: Detect Emotion and Output Results

* Displays:

  * Detected **Emotion**
  * **Transcription** of the audio
  * Detected **Language**

---

## 🖼️ Example

```python
audio_path = "/content/happy voice.mp4"
emotion_aware_speech_recognition(audio_path)
```

Output:

```
Detected Emotion: happy
Transcription: Hello, how are you?
Language Detected: en
```

---

## 📌 Notes

* The classifier in this notebook is trained on **randomly generated data**. You should replace it with real labeled emotion audio features for actual usage.
* Whisper is highly accurate but requires a good GPU for faster transcription.
* The notebook is structured to allow easy extension into a web app or API service.

---

## 🔮 Future Work

* Train on real emotion-labeled datasets (e.g., RAVDESS, TESS).
* Add multilingual emotion datasets.
* Extend to real-time audio (microphone input).
* Deploy using Streamlit or Flask for UI.

---
