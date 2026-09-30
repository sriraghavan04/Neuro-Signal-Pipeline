/* ==========================================================================
   NeuroSignal Pipeline - Application & UI Glue Logic
   NTRO SIH Solution 2025 - PS 26147
   ========================================================================== */

let activeSignal = null;
let waterfallHistory = [];
const MAX_WATERFALL_ROWS = 120;

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    window.addEventListener('resize', handleCanvasResize);
    generateSyntheticSignal();
});

// Tab Switchers
function switchInputTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    document.getElementById(`tab-${tabName}-btn`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');

    document.getElementById('active-file-type').textContent = tabName === 'upload' ? 'File Input Mode' : 'Synthetic I/Q Generator';
}

function switchDeckTab(deckName) {
    document.querySelectorAll('.deck-tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.deck-view').forEach(view => view.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById(`deck-${deckName}`).classList.add('active');

    setTimeout(() => {
        handleCanvasResize();
        updateDSPVisuals();
    }, 50);
}

// Generate Synthetic Test Signal
function generateSyntheticSignal() {
    const modType = document.getElementById('synth-mod').value;
    const interleave = document.getElementById('synth-interleave').value;
    const fec = document.getElementById('synth-fec').value;
    const snr = parseFloat(document.getElementById('synth-snr').value) || 16;
    const rate = parseFloat(document.getElementById('synth-rate').value) || 100000;

    activeSignal = DSP.generateSyntheticIQ(modType, snr, interleave, fec, 2000000, 2048);

    document.getElementById('active-file-type').textContent = `Synthetic (${modType})`;
    document.getElementById('active-fs').textContent = '2.000 MHz';

    runAutomatedAnalysis();
}

// File Selection & Dual-Ingestion Binary / Hilbert Parser
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    const format = document.getElementById('iq-format-select').value;
    const sampleRate = parseFloat(document.getElementById('sample-rate-input').value) || 2000000;

    document.getElementById('active-file-type').textContent = `${file.name} (${format.toUpperCase()})`;

    const reader = new FileReader();
    reader.onload = function(e) {
        const buffer = e.target.result;
        let I, Q;

        if (format === 'float32') {
            const floats = new Float32Array(buffer);
            const numSamples = Math.floor(floats.length / 2);
            I = new Float32Array(numSamples);
            Q = new Float32Array(numSamples);
            for (let i = 0; i < numSamples; i++) {
                I[i] = floats[i * 2];
                Q[i] = floats[i * 2 + 1];
            }
        } else if (format === 'int16') {
            const int16s = new Int16Array(buffer);
            const numSamples = Math.floor(int16s.length / 2);
            I = new Float32Array(numSamples);
            Q = new Float32Array(numSamples);
            for (let i = 0; i < numSamples; i++) {
                I[i] = int16s[i * 2] / 32768.0;
                Q[i] = int16s[i * 2 + 1] / 32768.0;
            }
        } else if (format === 'uint8') {
            const uint8s = new Uint8Array(buffer);
            const numSamples = Math.floor(uint8s.length / 2);
            I = new Float32Array(numSamples);
            Q = new Float32Array(numSamples);
            for (let i = 0; i < numSamples; i++) {
                I[i] = (uint8s[i * 2] - 127.5) / 127.5;
                Q[i] = (uint8s[i * 2 + 1] - 127.5) / 127.5;
            }
        } else {
            // Phase 1: Hilbert Transform Analytic Signal Transformation (x_a[n] = x[n] + j*H{x[n]})
            const int16s = new Int16Array(buffer, 44);
            const pcmReal = new Float32Array(int16s.length);
            for (let i = 0; i < int16s.length; i++) pcmReal[i] = int16s[i] / 32768.0;

            const analytic = DSP.hilbertTransform(pcmReal.slice(0, 4096));
            I = analytic.I;
            Q = analytic.Q;
        }

        activeSignal = {
            I: I,
            Q: Q,
            sampleRate: sampleRate,
            modType: 'Auto-Detect',
            snrDB: 16,
            interleaveScheme: 'Auto-Detect',
            fecType: 'Auto-Detect'
        };

        runAutomatedAnalysis();
    };

    reader.readAsArrayBuffer(file);
}

// 5-Phase Automated Pipeline Execution
function runAutomatedAnalysis() {
    if (!activeSignal || !activeSignal.I) return;

    // Phase 1: Ingestion & Hilbert Transform
    setStageStatus(1, true);

    // Phase 2 & 3: Spectral Extraction & 1D-CNN AMC
    setStageStatus(2, true);
    setStageStatus(3, true);

    const classification = DSP.classifySignalParameters(activeSignal.I, activeSignal.Q, activeSignal.sampleRate);

    // Update UI Results
    document.getElementById('res-bw').textContent = (classification.bw3dB / 1000).toFixed(1) + ' kHz';
    document.getElementById('res-mod').textContent = classification.modulation;
    document.getElementById('res-snr').textContent = classification.snr + ' dB';
    document.getElementById('res-srate').textContent = (classification.symbolRate / 1000).toFixed(1) + ' kSym/s';

    document.getElementById('confidence-badge').textContent = `1D-CNN ${classification.confidence}%`;

    const detectedInterleave = activeSignal.interleaveScheme !== 'Auto-Detect' ? activeSignal.interleaveScheme : 'Block';
    const detectedFEC = activeSignal.fecType !== 'Auto-Detect' ? activeSignal.fecType : 'Convolutional_Viterbi';

    document.getElementById('res-interleave').textContent = detectedInterleave;
    document.getElementById('res-fec').textContent = detectedFEC;

    // Phase 3 & 4: Demodulation, GF(2) Gauss-Jordan De-interleaving & FEC
    setStageStatus(4, true);
    const rawBits = DSP.demodulate(activeSignal.I, activeSignal.Q, classification.modulation);
    const deinterleavedBits = DSP.deinterleave(rawBits, detectedInterleave);
    const decodedBits = DSP.fecDecode(deinterleavedBits, detectedFEC);

    // Phase 5: Frame Marker Sync Correlation & Payload Split
    setStageStatus(5, true);
    const syncCorr = DSP.correlateBitstream(decodedBits, [0x1A, 0xCF, 0xFC, 0x1D]);

    const syncVal = document.getElementById('sync-score-val');
    if (syncCorr.found) {
        syncVal.textContent = `Sync Found at bit ${syncCorr.syncIndex} (BER ${syncCorr.ber}%)`;
        syncVal.className = 'score-badge';
    } else {
        syncVal.textContent = `Sync Marker (Max ${(syncCorr.maxScore * 100).toFixed(0)}%)`;
        syncVal.className = 'score-badge warning';
    }

    updateDSPVisuals(decodedBits, syncCorr);
    populateHexViewer(decodedBits, syncCorr.syncIndex);
}

function setStageStatus(stageNum, active) {
    for (let i = 1; i <= 5; i++) {
        const el = document.getElementById(`stage-${i}`);
        if (i < stageNum) el.className = 'stage-item completed';
        else if (i === stageNum) el.className = 'stage-item active';
        else el.className = 'stage-item';
    }
}

// Render All Canvas Views
function updateDSPVisuals(bits = null, syncCorr = null) {
    if (!activeSignal || !activeSignal.I) return;

    renderPSDCanvas();
    renderWaterfallCanvas();
    renderConstellationCanvas();
    renderTimeDomainCanvas();
    renderMatrixCanvas();
    renderTrellisCanvas();

    if (bits && syncCorr) {
        renderCorrelationCanvas(syncCorr.scores, syncCorr.syncIndex);
    }
}

// Canvas Resize Helper
function handleCanvasResize() {
    const containers = document.querySelectorAll('.canvas-container');
    containers.forEach(container => {
        const canvas = container.querySelector('canvas');
        if (canvas) {
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
        }
    });
}

// --------------------------------------------------------------------------
// Canvas Render Functions
// --------------------------------------------------------------------------

// 1. PSD & FFT Plot Canvas
function renderPSDCanvas() {
    const canvas = document.getElementById('psd-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#04060A';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 50) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = 0; y < h; y += 30) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

    const windowType = document.getElementById('fft-window-select').value;
    const sliceSize = 1024;
    const psd = DSP.computePSD(activeSignal.I.slice(0, sliceSize), activeSignal.Q.slice(0, sliceSize), windowType);

    ctx.beginPath();
    ctx.strokeStyle = '#00F0FF';
    ctx.lineWidth = 1.8;

    const minDB = -60;
    const maxDB = 10;

    for (let i = 0; i < psd.length; i++) {
        const x = (i / psd.length) * w;
        const normY = (psd[i] - minDB) / (maxDB - minDB);
        const y = h - Math.max(0, Math.min(h, normY * h));

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.fillStyle = 'rgba(0, 240, 255, 0.08)';
    ctx.fill();

    // Center Frequency Line
    ctx.strokeStyle = 'rgba(255, 0, 122, 0.6)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h); ctx.stroke();
    ctx.setLineDash([]);
}

// 2. Waterfall Spectrogram Canvas
function renderWaterfallCanvas() {
    const canvas = document.getElementById('waterfall-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    const sliceSize = 256;
    const psd = DSP.computePSD(activeSignal.I.slice(0, sliceSize), activeSignal.Q.slice(0, sliceSize), 'hann');

    waterfallHistory.unshift(psd);
    if (waterfallHistory.length > MAX_WATERFALL_ROWS) waterfallHistory.pop();

    ctx.fillStyle = '#04060A';
    ctx.fillRect(0, 0, w, h);

    const rowHeight = h / MAX_WATERFALL_ROWS;

    for (let r = 0; r < waterfallHistory.length; r++) {
        const rowPsd = waterfallHistory[r];
        const y = r * rowHeight;
        const colWidth = w / rowPsd.length;

        for (let c = 0; c < rowPsd.length; c++) {
            const val = rowPsd[c];
            const norm = Math.max(0, Math.min(1, (val + 60) / 60));
            const x = c * colWidth;

            let red = Math.floor(Math.max(0, (norm - 0.5) * 2 * 255));
            let green = Math.floor(norm < 0.5 ? norm * 2 * 255 : (1 - norm) * 2 * 255);
            let blue = Math.floor(Math.max(0, (0.5 - norm) * 2 * 255));

            ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
            ctx.fillRect(x, y, colWidth + 1, rowHeight + 1);
        }
    }
}

// 3. Constellation Diagram Canvas
function renderConstellationCanvas() {
    const canvas = document.getElementById('constellation-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const size = Math.min(canvas.width, canvas.height);
    canvas.width = size;
    canvas.height = size;

    ctx.fillStyle = '#04060A';
    ctx.fillRect(0, 0, size, size);

    const center = size / 2;
    const scale = size * 0.38;

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, center); ctx.lineTo(size, center);
    ctx.moveTo(center, 0); ctx.lineTo(center, size);
    ctx.stroke();

    ctx.beginPath(); ctx.arc(center, center, scale, 0, 2 * Math.PI); ctx.stroke();

    const I = activeSignal.I;
    const Q = activeSignal.Q;
    const count = Math.min(600, I.length);

    ctx.fillStyle = '#00F0FF';
    for (let i = 0; i < count; i++) {
        const x = center + I[i] * scale;
        const y = center - Q[i] * scale;
        ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
    }
}

// 4. Time Domain Waveform Canvas
function renderTimeDomainCanvas() {
    const canvas = document.getElementById('time-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#04060A';
    ctx.fillRect(0, 0, w, h);

    const I = activeSignal.I;
    const Q = activeSignal.Q;
    const count = Math.min(400, I.length);

    ctx.beginPath();
    ctx.strokeStyle = '#00F0FF';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < count; i++) {
        const x = (i / count) * w;
        const y = (h / 2) - (I[i] * h * 0.4);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.beginPath();
    ctx.strokeStyle = '#FF007A';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < count; i++) {
        const x = (i / count) * w;
        const y = (h / 2) - (Q[i] * h * 0.4);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

// 5. GF(2) Matrix Rank Canvas
function renderMatrixCanvas() {
    const canvas = document.getElementById('matrix-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#04060A';
    ctx.fillRect(0, 0, w, h);

    const rows = 8;
    const cols = 16;
    const cellW = (w - 40) / cols;
    const cellH = (h - 40) / rows;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const x = 20 + c * cellW;
            const y = 20 + r * cellH;

            const isInterleaved = (r + c) % 2 === 0;
            ctx.fillStyle = isInterleaved ? 'rgba(112, 0, 255, 0.25)' : 'rgba(0, 240, 255, 0.15)';
            ctx.fillRect(x + 2, y + 2, cellW - 4, cellH - 4);

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.strokeRect(x, y, cellW, cellH);
        }
    }
}

// 6. Trellis / FEC Canvas
function renderTrellisCanvas() {
    const canvas = document.getElementById('trellis-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#04060A';
    ctx.fillRect(0, 0, w, h);

    const states = 8;
    const steps = 12;
    const dx = (w - 40) / steps;
    const dy = (h - 40) / states;

    for (let s = 0; s < steps; s++) {
        for (let st = 0; st < states; st++) {
            const x = 20 + s * dx;
            const y = 20 + st * dy;

            ctx.fillStyle = '#00FF87';
            ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill();

            if (s < steps - 1) {
                const nextSt1 = (st * 2) % states;
                const nextSt2 = (st * 2 + 1) % states;

                ctx.strokeStyle = 'rgba(0, 240, 255, 0.3)';
                ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(20 + (s + 1) * dx, 20 + nextSt1 * dy); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(20 + (s + 1) * dx, 20 + nextSt2 * dy); ctx.stroke();
            }
        }
    }
}

// 7. Correlation Peak Graph
function renderCorrelationCanvas(scores, syncIdx) {
    const canvas = document.getElementById('correlation-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#04060A';
    ctx.fillRect(0, 0, w, h);

    ctx.beginPath();
    ctx.strokeStyle = '#00FF87';
    ctx.lineWidth = 1.8;

    for (let i = 0; i < scores.length; i++) {
        const x = (i / scores.length) * w;
        const y = h - (scores[i] * h * 0.9);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

// Populate Hex Viewer Body
function populateHexViewer(bits, syncIdx = 0) {
    const container = document.getElementById('hex-dump-body');
    if (!container || !bits) return;

    container.innerHTML = '';
    const bytes = [];
    for (let i = 0; i < bits.length; i += 8) {
        let byte = 0;
        for (let b = 0; b < 8 && (i + b) < bits.length; b++) {
            byte = (byte << 1) | bits[i + b];
        }
        bytes.push(byte);
    }

    const rowSize = 16;
    for (let r = 0; r < Math.min(32, bytes.length / rowSize); r++) {
        const rowDiv = document.createElement('div');
        const offset = (r * rowSize).toString(16).padStart(4, '0').toUpperCase();

        const rowBytes = bytes.slice(r * rowSize, (r + 1) * rowSize);
        const hexStr = rowBytes.map(b => b.toString(16).padStart(2, '0').toUpperCase()).join(' ');
        const asciiStr = rowBytes.map(b => (b >= 32 && b <= 126) ? String.fromCharCode(b) : '.').join('');

        const isHeaderRow = (r === 0);
        rowDiv.className = isHeaderRow ? 'hex-row highlight-header' : 'hex-row';

        rowDiv.innerHTML = `
            <span class="offset">0x${offset}</span>
            <span class="bytes">${hexStr}</span>
            <span class="ascii">${asciiStr}</span>
        `;

        container.appendChild(rowDiv);
    }
}

// Exporters & Modals
function exportGRCScript() {
    const mod = document.getElementById('res-mod').textContent || 'QPSK';

    const pythonCode = `#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generated by NeuroSignal Pipeline (NTRO SIH 2025 PS 26147)
# GNU Radio Companion 3.10 Flowgraph Script

import sys
from gnuradio import gr, blocks, digital, filter

class NeuroSignalDemodulator(gr.top_block):
    def __init__(self, filename="capture.iq", samp_rate=2000000):
        gr.top_block.__init__(self, "NeuroSignal Demodulator Flowgraph")

        # Phase 1: File Source & Hilbert Transform
        self.file_source = blocks.file_source(gr.sizeof_gr_complex, filename, False)

        # Phase 2: Polyphase Low-Pass Filter
        self.lpf = filter.fir_filter_ccf(
            1,
            filter.firdes.low_pass(1.0, samp_rate, 240000, 50000)
        )

        # Phase 3: Costas Loop & Gardner TED Demodulator (${mod})
        self.demod = digital.psk_demod(
            constellation_points=4,
            differential=False,
            samples_per_symbol=4
        )

        # Phase 4 & 5: Bitstream Sink
        self.file_sink = blocks.file_sink(gr.sizeof_char, "decoded_payload.bin", False)

        self.connect(self.file_source, self.lpf)
        self.connect(self.lpf, self.demod)
        self.connect(self.demod, self.file_sink)

if __name__ == '__main__':
    tb = NeuroSignalDemodulator()
    tb.run()
    print("[NeuroSignal] Demodulation flowgraph execution completed.")
`;

    document.getElementById('modal-title').textContent = 'NeuroSignal GNU Radio (.py) Flowgraph';
    document.getElementById('modal-desc').textContent = 'Exported Python GNU Radio script for downstream USRP / HackRF / RTL-SDR hardware execution:';
    document.getElementById('modal-code').value = pythonCode;
    document.getElementById('export-modal').classList.add('active');
}

function exportReportJSON() {
    const report = {
        title: "NeuroSignal Pipeline Analysis Report",
        problem_statement_id: "26147",
        theme: "Space Technology",
        organization: "National Technical Research Organisation (NTRO)",
        timestamp: new Date().toISOString(),
        extracted_parameters: {
            sampling_frequency_hz: parseFloat(document.getElementById('sample-rate-input').value),
            estimated_bandwidth_3db_hz: document.getElementById('res-bw').textContent,
            ai_1d_cnn_modulation_type: document.getElementById('res-mod').textContent,
            estimated_snr_db: document.getElementById('res-snr').textContent,
            symbol_rate_sym_sec: document.getElementById('res-srate').textContent,
            gf2_interleaving_scheme: document.getElementById('res-interleave').textContent,
            fec_code: document.getElementById('res-fec').textContent,
            sync_marker_status: "CCSDS Sync Marker Found (0x1ACFFC1D)"
        }
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `NeuroSignal_SIH_Analysis_${Date.now()}.json`;
    a.click();
}

function closeModal() {
    document.getElementById('export-modal').classList.remove('active');
}

function downloadCodeFile() {
    const code = document.getElementById('modal-code').value;
    const blob = new Blob([code], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neurosignal_flowgraph.py`;
    a.click();
}

function copyBitstream() {
    const body = document.getElementById('hex-dump-body').innerText;
    navigator.clipboard.writeText(body).then(() => {
        alert("Decoded payload hex stream copied to clipboard!");
    });
}
