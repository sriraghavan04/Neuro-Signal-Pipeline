# 🧠 NeuroSignal Pipeline

> **Automated model for analysis of .IQ and .wav files along with signal parameter extraction**  
> **Smart India Hackathon 2025 (PS ID: 26147)**  
> **Theme:** Space Technology | **Organization:** National Technical Research Organisation (NTRO)

---

## 📌 Overview

**NeuroSignal Pipeline** is a software-defined radio (SDR) and digital signal processing (DSP) intelligence platform built for blind parameter extraction, modulation classification, de-interleaving, FEC decoding, and frame payload parsing from complex baseband (`.IQ`) and real-valued IF (`.wav`) signals.

---

## ✨ Features & 5-Phase Architecture

### 1. Phase 1: Dual-Ingestion & Pre-Processing
- **Complex Baseband (`.IQ`)**: Supports Float32, Int16, and UInt8 raw I/Q data.
- **Real-Valued IF Audio (`.wav`)**: Hilbert Transform analytic signal conversion ($x_a[n] = x[n] + j\mathcal{H}\{x[n]\}$).
- **Filtering**: Adaptive Wiener noise reduction filter.

### 2. Phase 2: Spectral Parameter Extraction
- **Welch's PSD & Bandwidth**: $-3\text{ dB}$ and $-10\text{ dB}$ threshold bandwidth calculation.
- **Cyclostationary Baud Rate ($R_s$)**: $|x[n]|^2$ FFT spectral line extraction.

### 3. Phase 3: Synchronization & 1D-CNN AMC
- **1D-CNN Automatic Modulation Classification**: Evaluates 4th-order cyclic cumulants ($C_{40}, C_{42}$) for BPSK, QPSK, 8-PSK, 16-QAM, and 2-FSK with confidence scores.
- **Gardner TED & Costas Loop**: Symbol timing error detection and carrier phase recovery.

### 4. Phase 4: Blind De-Interleaving & FEC Reconstruction
- **$\text{GF}(2)$ Gauss-Jordan Matrix Rank Analysis**: Mathematical deduction of unknown interleaver depth and parity-check matrix $H$ structures.
- **FEC Decoders**: Viterbi Convolutional Code ($r=1/2$, $K=7$), Reed-Solomon RS(15,11), and LDPC.

### 5. Phase 5: Frame Correlation & Payload Parsing
- **Sync Marker Correlation**: CCSDS space telemetry preamble search (`0x1ACFFC1D`).
- **Header & Payload Splitter**: Hex and ASCII payload inspection.

---

## 🛠️ Project Structure

```
.
├── index.html       # Single Page Application structure & UI deck tabs
├── index.css        # Space-tech dark obsidian theme & glassmorphism styling
├── dsp.js           # Core DSP, Hilbert Transform, FFT, AMC, De-interleaver & FEC engine
└── app.js           # File loader, Canvas renders & GRC export glue
```

---

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/sriraghavan04/Neuro-Signal-Pipeline.git
   cd Neuro-Signal-Pipeline
   ```

2. Run local server:
   ```bash
   python -m http.server 8085
   ```

3. Open `http://localhost:8085` in your browser.
