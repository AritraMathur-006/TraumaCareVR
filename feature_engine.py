import os
import shutil
import imageio_ffmpeg

# 1. Register imageio-ffmpeg executable
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe)
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

# 2. Copy ffmpeg.exe directly into current pipeline folder if not present
local_ffmpeg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
if not os.path.exists(local_ffmpeg):
    try:
        shutil.copyfile(ffmpeg_exe, local_ffmpeg)
        print(f"[ENGINE] Mirrored ffmpeg.exe locally to: {local_ffmpeg}")
    except Exception:
        pass

# 3. Patch Whisper's internal audio loader to call ffmpeg_exe directly
import whisper.audio
whisper.audio.FFMPEG_BINARY = ffmpeg_exe
print(f"[ENGINE] Hardcoded Whisper FFMPEG_BINARY -> {ffmpeg_exe}")

import re
import difflib
import numpy as np
import pandas as pd
import librosa
import joblib
import soundfile as sf
import whisper
from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else r"D:\SIH26094_PIPELINE"
MODEL_PATH = os.path.join(BASE_DIR, "distress_model.joblib")

# Load pre-trained models
print("[ENGINE] Loading RoBERTa Sentiment Pipeline...")
nlp_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", truncation=True, max_length=512)

print("[ENGINE] Loading Whisper STT Engine...")
whisper_model = whisper.load_model("base")  # Fast inference for real-time triage

# Comprehensive Multilingual & Transliterated Trauma Lexicon
TRAUMA_KEYWORDS = [
    # Severe Threat & Physical Harm (English + Hindi/Hinglish)
    "threat", "threaten", "kill", "murder", "harm", "attack", "assault", "stab", "gun", "pistol",
    "goli", "maar", "marne", "kat", "chaku", "laathi", "pitayi", "jaan", "khatra", "dhamki", 
    "dhamkana", "barbaad", "tabah", "raktdan",

    # Psychological Distress, Fear & Panic
    "fear", "afraid", "scared", "terrified", "panic", "anxiety", "nervous", "cry", "tremble",
    "darr", "dar", "sehmi", "ghabrahat", "chinta", "pareshaan", "behosh", "kaamp", "rona", 
    "darawana", "ashanti", "dastak",

    # Suicide & Acute Hopelessness
    "suicide", "hopeless", "depressed", "worthless", "end", "die", "death", "poison",
    "aatmhatya", "marjana", "zahar", "khatam", "mout", "lash", "barbad",

    # Coercion, Stalking, Surveillance & Intimidation
    "follow", "stalk", "kidnap", "hostage", "chase", "watch", "record", "extort", "bribe", 
    "picha", "ghera", "uthana", "agwah", "paise", "vasooli", "dabaav", "blackmail", "zabardasti",

    # Legal, Judicial & Procedural Pressure
    "court", "police", "fir", "jail", "advocate", "judge", "testify", "witness", "statement", 
    "bayan", "gavah", "tareekh", "peshi", "kacheri", "thaana", "daroga", "vakil", "boycott",
    "samaaj", "ghar", "unsafe", "alone", "isolate", "akela"
]

def transcribe_audio(audio_path: str) -> str:
    """
    Robust STT pipeline tuned for Indian English, Hindi, and Hinglish.
    Uses balanced decoding parameters to prevent hallucinations and garbled words.
    """
    try:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            print("[STT WARNING] Audio file is missing or 0 bytes.")
            return ""
            
        print(f"[STT INGEST] Processing audio: {audio_path}")
        
        # A natural bilingual prompt that anchors vocabulary for BOTH English and Hinglish
        balanced_prompt = (
            "Witness statement in court. Threat to life, fear, darr, dhamki, "
            "police FIR, emergency help, someone is following me."
        )
        
        # Do NOT force language="en" so it doesn't distort natural phonemes.
        # temperature=0.0 makes transcription deterministic and stops hallucinations.
        result = whisper_model.transcribe(
            audio_path,
            fp16=False,
            temperature=0.0,
            beam_size=5,
            best_of=5,
            initial_prompt=balanced_prompt
        )
        
        transcript = result.get("text", "").strip()
        print(f"[STT DETECTED] \"{transcript}\"")
        return transcript
        
    except Exception as e:
        print(f"[STT FAILED] Whisper transcription error: {e}")
        return ""
def extract_audio_features(audio_path: str):
    """Zero-retention acoustic feature extraction with waveform arrays."""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"[ACOUSTIC LOAD ERROR] Fallback to synthetic frame: {e}")
        sr = 16000
        y = np.zeros(sr * 2)

    # Pitch tracking (F0 variance / Jitter proxy)
    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        
        if len(pitch_values) > 1:
            jitter = float(np.mean(np.abs(np.diff(pitch_values))) / (np.mean(pitch_values) + 1e-6))
        else:
            jitter = 0.03
            pitch_values = np.array([220.0, 225.0, 218.0])
    except Exception:
        jitter = 0.03
        pitch_values = np.array([220.0, 225.0, 218.0])
        
    jitter = float(np.clip(jitter, 0.01, 0.15))
    
    # Silence / hesitation estimation
    try:
        intervals = librosa.effects.split(y, top_db=25)
        non_silent = sum([end - start for start, end in intervals])
        total = len(y)
        pause_ratio = float((total - non_silent) / total) if total > 0 else 0.1
    except Exception:
        pause_ratio = 0.1
        
    pause_ratio = float(np.clip(pause_ratio, 0.05, 0.60))
    
    return jitter, pause_ratio, y, sr, pitch_values[:200]

def extract_text_features(text: str):
    """
    Standardized, deterministic keyword extraction that treats typed and 
    spoken text with identical semantic mapping.
    """
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower()).strip()
    words = clean_text.split()
    
    matched_keywords = set()

    for word in words:
        # Collapse repeated characters (e.g., "darrrr" -> "darr", "kiiiill" -> "kill")
        norm_word = re.sub(r'(.)\1{2,}', r'\1\1', word)
        
        # 1. Direct or stem match against the trauma dictionary
        for kw in TRAUMA_KEYWORDS:
            if kw == norm_word or (len(kw) > 3 and kw in norm_word):
                matched_keywords.add(kw)
                break
        else:
            # 2. Fuzzy match for typos if no direct stem matched (min length 4 to avoid false matches)
            if len(norm_word) >= 4:
                matches = difflib.get_close_matches(norm_word, TRAUMA_KEYWORDS, n=1, cutoff=0.82)
                if matches:
                    matched_keywords.add(matches[0])

    keyword_count = len(matched_keywords)
    if matched_keywords:
        print(f"[NLP DETECTED KEYWORDS] Unique matches ({keyword_count}): {list(matched_keywords)}")

    # RoBERTa sentiment evaluation
    try:
        if not clean_text or clean_text == "statement not provided":
            neg_sentiment = 0.5
        else:
            analysis = nlp_pipeline(clean_text)[0]
            label = analysis['label'].lower()
            score = analysis['score']
            
            if 'negative' in label:
                neg_sentiment = float(score)
            elif 'positive' in label:
                neg_sentiment = float(1.0 - score)
            else:
                neg_sentiment = 0.5
    except Exception as e:
        print(f"[NLP ERROR] RoBERTa failed, fallback to neutral: {e}")
        neg_sentiment = 0.5
        
    return keyword_count, neg_sentiment

def compute_distress_score(audio_path: str, transcript: str, days_to_hearing=10, intimidation_flag=0):
    """End-to-end multimodal distress assessment with deterministic critical safeguards."""
    model = joblib.load(MODEL_PATH)
    
    # 1. Acoustic telemetry extraction
    jitter, pause, y, sr, sample_pitches = extract_audio_features(audio_path)
    
    # 2. Automated STT fallback if transcript is empty
    if not transcript or transcript.strip() == "":
        transcript = transcribe_audio(audio_path)
        
    if not transcript or transcript.strip() == "":
        transcript = "Statement not provided."
        
    # 3. Linguistic & sentiment extraction
    keywords, neg_sentiment = extract_text_features(transcript)
    
    # 4. Feature vector construction
    features = pd.DataFrame([{
        'pitch_jitter': jitter,
        'pause_ratio': pause,
        'trauma_keywords': keywords,
        'negative_sentiment': neg_sentiment,
        'days_to_hearing': days_to_hearing,
        'intimidation_flag': intimidation_flag
    }])
    
    score = float(model.predict(features)[0])
    
    # 5. Critical crisis override safeguard
    # Active threats or multiple trauma markers trigger emergency escalation directly
    if intimidation_flag == 1 and keywords >= 2:
        score = max(score, 82.5)
    elif intimidation_flag == 1 or keywords >= 3:
        score = max(score, 74.0)

    score = float(np.clip(score, 5.0, 98.0))
    
    # 6. Triage categorization
    if score < 40:
        triage = "GREEN (Stable)"
    elif score < 70:
        triage = "YELLOW (Elevated Distress)"
    else:
        triage = "RED (Critical / Acute Crisis)"
        
    return {
        "score": round(score, 2),
        "triage": triage,
        "transcript": transcript,
        "raw_audio": y,
        "sample_rate": sr,
        "pitches": sample_pitches,
        "features": {
            "pitch_jitter": round(jitter, 4),
            "pause_ratio": round(pause, 4),
            "trauma_keywords": keywords,
            "negative_sentiment": round(neg_sentiment, 4)
        }
    }