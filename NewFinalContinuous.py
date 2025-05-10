import sounddevice as sd
import numpy as np
import wave
import time
import librosa
import tensorflow as tf
import os

# Set environment variables for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Audio parameters
SAMPLE_RATE = 16000
DURATION = 2
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
N_MFCC = 20
MAX_PAD_LEN = 32
OUTPUT_DIR = r"/content/drive/MyDrive/soundclip"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "recorded_clip.wav")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def record_audio(duration, sample_rate):
    print("🎤 Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    print("✅ Recording complete!")
    return audio_data

def save_audio(filename, audio_data, sample_rate):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    print(f"📁 Audio saved as: {filename}")

def extract_features(file_path, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        window_size = 16000  # 1 second to match training
        features_list = []
        for start in range(0, len(audio), window_size):
            end = start + window_size
            audio_window = audio[start:end]
            if len(audio_window) < window_size:
                audio_window = np.pad(audio_window, (0, window_size - len(audio_window)), "constant")
            mfcc = librosa.feature.mfcc(y=audio_window, sr=sr, n_mfcc=n_mfcc)
            if mfcc.shape[1] < max_pad_len:
                pad_width = max_pad_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
            else:
                mfcc = mfcc[:, :max_pad_len]
            features_list.append(mfcc)
        return np.array(features_list)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def predict_wake_word(model, file_path, threshold=0.4):
    features = extract_features(file_path)
    if features is None:
        print("Failed to extract features.")
        return None

    features = features[..., np.newaxis]
    try:
        predictions = model.predict(features)
        scores = predictions.flatten()
        max_score = np.max(scores)
        if max_score > threshold:
            print(f"Wake word detected! (Score: {max_score:.2f})")
            time.sleep(5)
        else:
            print(f"No wake word. (Score: {max_score:.2f})")
        return max_score
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

if __name__ == "__main__":
    try:
        model = tf.keras.models.load_model(r"C:\Users\aryan\Downloads\python_wake_word_model_improved.h5")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    try:
        while True:
            recorded_audio = record_audio(DURATION, SAMPLE_RATE)
            save_audio(OUTPUT_FILE, recorded_audio, SAMPLE_RATE)
            print("\nAnalyzing the recorded clip...")
            predict_wake_word(model, OUTPUT_FILE)
            print("Starting next recording cycle...")
    except KeyboardInterrupt:
        print("\nStopping the recording and prediction loop.")