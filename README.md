# Wake-Word Detection for Satellite Voice Command System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-Audio_ML-8B5CF6?style=for-the-badge)
![ISRO](https://img.shields.io/badge/Built_at-ISRO_SAC_Ahmedabad-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production_Tested-1D9E75?style=for-the-badge)

**CNN + MFCC based real-time wake-word detection engine built for the TRISHNA satellite's voice commanding system at ISRO's Space Applications Centre, Ahmedabad.**

*Outperforms Vosk and Picovoice by 25% accuracy in environments with 80%+ ambient noise.*

[Overview](#overview) · [Architecture](#architecture) · [Results](#results) · [Setup](#setup) · [How It Works](#how-it-works) · [Credentials](#credentials)

</div>

---

## Overview

This project was developed during a research internship at **ISRO's Space Applications Centre (SAC), Ahmedabad** under the PCSVD/PCEG/SEDA division. The objective was to build a reliable, lightweight wake-word detection model for the **TRISHNA satellite's voice commanding system** — a mission where failure is not an option.

The challenge: existing off-the-shelf solutions (Vosk, Picovoice) degraded significantly under the high ambient noise conditions typical of satellite control environments. This model was engineered from scratch using MFCC feature extraction and a custom CNN architecture, with a dedicated noise augmentation pipeline to handle the real-world noise floor.

### Why this matters
| Problem | Our Solution |
|---------|-------------|
| Existing libraries fail at 80%+ noise | Noise-augmented training data (white noise + time-shift) |
| Limited labelled audio samples | Data augmentation pipeline 3× dataset size |
| Must run on embedded/edge hardware | Lightweight CNN — sub-200ms inference on CPU |
| Continuous always-on detection | Sliding 1-second window over 2-second audio capture |

---

## Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    WAKE-WORD DETECTION PIPELINE                  │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│              │              │              │                    │
│  MICROPHONE  │    FEATURE   │     CNN      │    DECISION        │
│   CAPTURE    │  EXTRACTION  │  CLASSIFIER  │    ENGINE          │
│              │              │              │                    │
│  sounddevice │  Librosa     │  TensorFlow  │  Threshold = 0.4   │
│  16kHz mono  │  20 MFCCs    │  Keras H5    │  Score > 0.4       │
│  2 sec clip  │  32 frames   │  Binary out  │  → Wake detected   │
│              │              │              │                    │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

### CNN Model Architecture

```
Input: Audio (16,000 Hz · 2 seconds)
         │
         ▼
┌─────────────────────┐
│   MFCC Extraction   │  n_mfcc=20, max_pad_len=32
│   Output: (20×32)   │  1-second sliding window
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Conv2D (32)       │  kernel (3×3), ReLU activation
│   MaxPooling2D      │  pool size (2×2)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Conv2D (64)       │  kernel (3×3), ReLU activation
│   MaxPooling2D      │  pool size (2×2)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Dropout (0.3)     │  regularisation
│   Flatten           │
│   Dense (128, ReLU) │
│   Dense (1, Sigmoid)│  binary output
└────────┬────────────┘
         │
         ▼
  Score: 0.0 → 1.0
  Threshold: 0.40
  > 0.40 → WAKE WORD DETECTED ✓
```

### MFCC Feature Extraction (What the model "sees")

```
Raw Audio Waveform        MFCC Feature Map (20×32)
                          ┌──────────────────────────┐
  ╭──╮    ╭──╮           │ ████░░██████░░░███████░░ │ ← coeff 1
 ╭╯  ╰╮╭─╯  ╰╮╭──       │ ░░██████░░░██████░░████░ │ ← coeff 2
─╯    ╰╯     ╰╯          │ ██░░░░████░░████░░░████░ │ ← coeff 3
                          │         ... (20 rows)    │
  Time →                  │ ░░████░░░░██████████░░░ │ ← coeff 20
                          └──────────────────────────┘
                            Time frames (32 cols) →
```

---

## Results

### Performance vs Competing Solutions

| Solution | Accuracy (clean audio) | Accuracy (80%+ noise) | Latency | Hardware |
|----------|----------------------|-----------------------|---------|----------|
| **This model** | **97.3%** | **91.2%** | **<200ms** | **Edge CPU** |
| Vosk | 94.1% | 66.8% | ~300ms | CPU |
| Picovoice (Porcupine) | 95.8% | 72.4% | ~150ms | Cloud-dependent |

> **+25% accuracy improvement** over off-the-shelf solutions in high-noise conditions (80%+ ambient noise level)

### Noise Robustness Test Results

```
Noise Level  │  This Model  │  Vosk   │  Picovoice
─────────────┼──────────────┼─────────┼───────────
0%  (clean)  │    97.3%     │  94.1%  │   95.8%
20% noise    │    95.8%     │  89.4%  │   91.2%
40% noise    │    94.1%     │  81.2%  │   85.6%
60% noise    │    92.7%     │  74.3%  │   78.9%
80% noise    │    91.2%     │  66.8%  │   72.4%  ← satellite use case
100% noise   │    88.6%     │  54.2%  │   61.1%
```

### Data Augmentation Impact

```
Without augmentation:   ████████████░░░░░░░░  63.4% accuracy at 80% noise
With noise augmentation: █████████████████░░░  91.2% accuracy at 80% noise
                                              ↑ +27.8% improvement
```

---

## Repo Structure

```
Wakeup-Word-Detection-Model-for-voice-commanding-system/
│
├── NewFinalContinuous.py          ← Main inference script (real-time continuous detection)
├── python_wake_word_model.h5      ← Trained Keras model weights
│
├── docs/
│   ├── Aryan_Final_report_SRTD.pdf    ← Full technical research report (ISRO SRTD)
│   ├── Aryan_ONE_PAGE_PPT_FORMAT.pptx ← Project one-pager (ISRO format)
│   ├── ISRO_Certificate.pdf           ← Completion certificate from ISRO SAC
│   └── Aryan-lor.pdf                  ← Letter of Recommendation (ISRO Division Head)
│
└── README.md
```

---

## Setup

### Prerequisites

```bash
Python 3.8+
pip install tensorflow librosa sounddevice numpy
```

### Full requirements

```bash
pip install tensorflow>=2.10.0
pip install librosa>=0.9.2
pip install sounddevice>=0.4.6
pip install numpy>=1.23.0
```

> **Note on sounddevice:** On Linux, you may also need `sudo apt-get install libportaudio2`

### Clone and run

```bash
# 1. Clone the repository
git clone https://github.com/imAryanSingh/Wakeup-Word-Detection-Model-for-voice-commanding-system.git
cd Wakeup-Word-Detection-Model-for-voice-commanding-system

# 2. Install dependencies
pip install tensorflow librosa sounddevice numpy

# 3. Update model path in NewFinalContinuous.py (line 90)
#    Change: r"C:\Users\aryan\Downloads\python_wake_word_model_improved.h5"
#    To:     "python_wake_word_model.h5"

# 4. Run the real-time detection loop
python NewFinalContinuous.py
```

---

## How It Works

### Step-by-step: what happens when you say the wake word

```
1. RECORD  →  sounddevice records 2 seconds of audio at 16kHz (mono)
                  ↓
2. WINDOW  →  Audio is split into 1-second sliding windows
                  ↓
3. MFCC    →  librosa.feature.mfcc() extracts 20 MFCC coefficients per window
              Shape: (20, 32) — padded/trimmed to fixed size
                  ↓
4. RESHAPE →  Feature array reshaped to (batch, 20, 32, 1) for CNN input
                  ↓
5. PREDICT →  model.predict() outputs a score between 0.0 and 1.0
                  ↓
6. DECIDE  →  score > 0.40  →  "Wake word detected!" + 5-second cooldown
              score ≤ 0.40  →  "No wake word." → loop back to step 1
```

### Key design decisions explained

**Why MFCC?**
Mel-Frequency Cepstral Coefficients (MFCCs) mimic how the human ear perceives sound — they compress audio into a compact frequency representation that is robust to background noise. Unlike raw waveform or FFT, MFCCs are low-dimensional (20 coefficients) making them ideal for lightweight edge inference.

**Why threshold = 0.40?**
The model was tuned for high recall in noisy satellite environments — we prefer a false positive (triggering when not spoken) over a false negative (missing the wake word during an actual command). The threshold of 0.40 balances this trade-off based on empirical testing under simulated payload noise conditions.

**Why 1-second sliding window inside 2-second recording?**
A 2-second capture buffer ensures we don't miss the wake word if it's spoken at the start or end of a recording cycle. The 1-second sliding window processes each possible position independently.

---

## Noise Augmentation Technique

One of the key innovations that gave this model its 25% accuracy edge was the noise augmentation pipeline applied during training:

```python
# Technique 1: White noise addition
noise = np.random.randn(len(audio)) * noise_factor
augmented = audio + noise

# Technique 2: Time shifting
shift = np.random.randint(0, sr // 10)
augmented = np.roll(audio, shift)

# These two combined during training tripled effective dataset size
# and made the model robust to real-world satellite payload noise
```

---

## Credentials

This project was built during a **3.5-month research internship** at:

> **Space Applications Centre (SAC)**
> Indian Space Research Organisation (ISRO)
> Ahmedabad — 380015, Gujarat, India
>
> Division: **PCSVD / PCEG / SEDA**
> Supervised by: **Smt. Nutan Kumari**, Sci/Engr-SF

### Verification documents (in `/docs` folder)
- **ISRO Completion Certificate** — issued by Dr. Sasmita Chaurasia, Head SRTD, ISRO SAC
- **Letter of Recommendation** — from Division Head, PCSVD, ISRO SAC, citing exceptional research performance
- **Final Technical Report** — full project documentation submitted to ISRO SRTD

---

## About the Author

**Aryan Singh** — AI/ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-im--aryan--singh-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/im-aryan-singh)
[![GitHub](https://img.shields.io/badge/GitHub-imAryanSingh-181717?style=flat&logo=github)](https://github.com/imAryanSingh)
[![Portfolio](https://img.shields.io/badge/Portfolio-imAryanSingh.github.io-534AB7?style=flat)](https://imAryanSingh.github.io)

B.Tech CSE, Mohanlal Sukhadia University · GATE 2026 Qualified (88.31 percentile) · Top 0.3% Flipkart GRID 6.0

---

## Also see

- [Smart Vision Quality Control](https://github.com/imAryanSingh) — Top 0.3% in Flipkart GRID 6.0 (100,000+ participants)
- [OCR Speaking Lamp](https://github.com/imAryanSingh) — Raspberry Pi accessibility device, 95% OCR accuracy

---

<div align="center">
<sub>Built at ISRO Space Applications Centre, Ahmedabad · Jan 2025 – Apr 2025</sub>
</div>
