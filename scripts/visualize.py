"""Visualization suite for good-noise benchmark results.

Generates publication-quality figures saved to results/:
  1. waveform_comparison.png   — clean vs protected waveforms (all 6 attacks)
  2. spectrogram_grid.png      — log-mel spectrograms: clean vs best 3 attacks
  3. metrics_radar.png         — radar chart of all 4 metrics per attack
  4. metrics_bar.png           — grouped bar chart: all metrics
  5. perturbation_detail.png   — zoomed waveform + FFT of the perturbation itself
"""

import sys, warnings, json
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import scipy.signal

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_DIR = Path("samples")
OUTPUTS_DIR = Path("outputs/benchmark")
SUMMARY_PATH = Path("experiments/benchmark_summary.json")

ATTACKS = [
    "spectral_filter",
    "mel_disruption",
    "pgd_embedding",
    "psychoacoustic_pgd",
    "universal_pgd",
    "asr_disruption",
]
ATTACK_LABELS = {
    "spectral_filter": "Spectral\nFilter",
    "mel_disruption": "Mel\nDisruption",
    "pgd_embedding": "PGD\nEmbedding",
    "psychoacoustic_pgd": "Psycho\nPGD",
    "universal_pgd": "Universal\nPGD",
    "asr_disruption": "ASR\nDisruption",
}

# Representative speaker/utterance for wave plots
REPR_SPEAKER = "1320"
REPR_UTTE = "utte00"

COLORS = {
    "spectral_filter": "#FF6B6B",
    "mel_disruption": "#FFD93D",
    "pgd_embedding": "#6BCB77",
    "psychoacoustic_pgd": "#4D96FF",
    "universal_pgd": "#C77DFF",
    "asr_disruption": "#FF9F43",
}

SR = 16000


def load_wav(path):
    wav, _ = sf.read(str(path), dtype="float32")
    return wav


def compute_mel(wav, sr=SR, n_fft=1024, hop=256, n_mels=80):
    """Compute log-mel spectrogram for visualization."""
    from scipy.signal import stft as scipy_stft
    import numpy as np

    f, t, Zxx = scipy_stft(wav, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Zxx)

    # Mel filterbank
    f_min, f_max = 80.0, 7600.0
    m_min = 2595 * np.log10(1 + f_min / 700)
    m_max = 2595 * np.log10(1 + f_max / 700)
    mel_pts = np.linspace(m_min, m_max, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, len(f) - 1)
    fbank = np.zeros((n_mels, len(f)))
    for m in range(1, n_mels + 1):
        s, c, e = bins[m-1], bins[m], bins[m+1]
        for k in range(s, c):
            if c > s: fbank[m-1, k] = (k - s) / (c - s)
        for k in range(c, e):
            if e > c: fbank[m-1, k] = (e - k) / (e - c)
    mel = fbank @ mag
    return np.log1p(mel * 1e4), t


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Waveform + perturbation comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_waveform_comparison():
    clean_path = SAMPLES_DIR / REPR_SPEAKER / f"{REPR_UTTE}.wav"
    clean = load_wav(clean_path)

    # Only use first 4 seconds for clarity
    n = min(len(clean), 4 * SR)
    clean = clean[:n]
    t = np.arange(n) / SR

    fig, axes = plt.subplots(len(ATTACKS) + 1, 2, figsize=(16, 14))
    fig.patch.set_facecolor("#0F0F1A")

    def style_ax(ax):
        ax.set_facecolor("#1A1A2E")
        ax.spines["bottom"].set_color("#444466")
        ax.spines["left"].set_color("#444466")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors="#888899", labelsize=7)

    # Clean row
    style_ax(axes[0, 0])
    axes[0, 0].plot(t, clean, color="#CCCCDD", linewidth=0.5, alpha=0.9)
    axes[0, 0].set_title("Clean Audio (LibriSpeech, Speaker 1320)", color="white", fontsize=9)
    axes[0, 0].set_ylabel("Amplitude", color="#888899", fontsize=7)
    axes[0, 0].set_xlim(0, 4)

    # Clean FFT
    style_ax(axes[0, 1])
    freqs = np.fft.rfftfreq(n, 1 / SR)
    fft = np.abs(np.fft.rfft(clean))
    axes[0, 1].semilogy(freqs / 1000, fft + 1e-6, color="#CCCCDD", linewidth=0.6, alpha=0.9)
    axes[0, 1].set_title("Frequency Spectrum", color="white", fontsize=9)
    axes[0, 1].set_xlabel("Frequency (kHz)", color="#888899", fontsize=7)
    axes[0, 1].set_xlim(0, 8)

    for i, aname in enumerate(ATTACKS):
        out_path = OUTPUTS_DIR / f"{aname}_spk{REPR_SPEAKER}_{REPR_UTTE}.wav"
        if not out_path.exists():
            continue
        protected = load_wav(out_path)[:n]
        color = COLORS[aname]

        # Protected waveform
        style_ax(axes[i + 1, 0])
        axes[i + 1, 0].plot(t, protected, color=color, linewidth=0.5, alpha=0.85)
        axes[i + 1, 0].set_ylabel(ATTACK_LABELS[aname].replace("\n", " "),
                                   color=color, fontsize=7.5, fontweight="bold")
        axes[i + 1, 0].set_xlim(0, 4)

        # Perturbation spectrum
        style_ax(axes[i + 1, 1])
        delta = protected - clean
        delta_fft = np.abs(np.fft.rfft(delta))
        axes[i + 1, 1].semilogy(freqs / 1000, delta_fft + 1e-9, color=color,
                                  linewidth=0.6, alpha=0.85, label="perturbation")
        axes[i + 1, 1].semilogy(freqs / 1000, fft + 1e-6, color="#444455",
                                  linewidth=0.4, alpha=0.5, label="clean")
        axes[i + 1, 1].set_xlim(0, 8)

        if i == len(ATTACKS) - 1:
            axes[i + 1, 1].set_xlabel("Frequency (kHz)", color="#888899", fontsize=7)

    fig.suptitle("good-noise: Clean vs. Protected Audio\n(LibriSpeech test-clean, waveforms and perturbation spectra)",
                 color="white", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout(h_pad=0.4)
    out = RESULTS_DIR / "waveform_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Log-mel spectrogram grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectrogram_grid():
    clean_path = SAMPLES_DIR / REPR_SPEAKER / f"{REPR_UTTE}.wav"
    clean = load_wav(clean_path)
    n = min(len(clean), 5 * SR)
    clean = clean[:n]

    show_attacks = ["pgd_embedding", "psychoacoustic_pgd", "mel_disruption", "universal_pgd"]
    ncols = len(show_attacks) + 1  # clean + 4 attacks

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5))
    fig.patch.set_facecolor("#0F0F1A")

    mel_clean, t_clean = compute_mel(clean)

    for ax, label, wav in zip(
        axes,
        ["Clean"] + [ATTACK_LABELS[a] for a in show_attacks],
        [clean] + [load_wav(OUTPUTS_DIR / f"{a}_spk{REPR_SPEAKER}_{REPR_UTTE}.wav")[:n]
                   for a in show_attacks],
    ):
        mel, _ = compute_mel(wav)
        vmin, vmax = mel_clean.min(), mel_clean.max() * 0.9

        im = ax.imshow(mel, aspect="auto", origin="lower", cmap="inferno",
                       vmin=vmin, vmax=vmax,
                       extent=[0, n / SR, 0, 80])
        ax.set_facecolor("#0F0F1A")
        ax.set_title(label, color="white", fontsize=9, fontweight="bold")
        ax.set_xlabel("Time (s)", color="#888899", fontsize=8)
        ax.tick_params(colors="#888899", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#444466")

    axes[0].set_ylabel("Mel band", color="#888899", fontsize=8)
    fig.suptitle("Log-Mel Spectrogram: Clean vs. Protected\n(Speaker 1320 — perturbation is imperceptible but spectral structure is disrupted)",
                 color="white", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "spectrogram_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Metrics bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_bar():
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    attacks = [r["attack"] for r in summary["results"]]
    snr   = [r["mean_snr"] for r in summary["results"]]
    pesq  = [r["mean_pesq"] for r in summary["results"]]
    stoi  = [r["mean_stoi"] for r in summary["results"]]
    sim   = [r["mean_speaker_similarity"] for r in summary["results"]]

    x = np.arange(len(attacks))
    width = 0.18
    short_labels = [ATTACK_LABELS[a].replace("\n", "\n") for a in attacks]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#0F0F1A")

    bar_data = [
        (axes[0], snr, "SNR (dB)", "#4D96FF", 20.0, True, "Higher is better\n(target: >20 dB)"),
        (axes[1], pesq, "PESQ", "#6BCB77", 3.5, True, "Higher is better\n(target: >3.5)"),
        (axes[2], stoi, "STOI", "#FFD93D", 0.90, True, "Higher is better\n(target: >0.90)"),
        (axes[3], sim, "Speaker Similarity", "#FF6B6B", 0.25, False, "Lower is better\n(target: <0.25)"),
    ]

    for ax, vals, ylabel, color, threshold, higher_better, subtitle in bar_data:
        ax.set_facecolor("#1A1A2E")
        colors = [COLORS[a] for a in attacks]
        bars = ax.bar(x, vals, width=0.6, color=colors, alpha=0.85, edgecolor="#333344")
        ax.axhline(threshold, color="white", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Target: {threshold}")
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=7.5, color="white")
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.set_title(subtitle, color="#888899", fontsize=8)
        ax.tick_params(colors="#888899")
        for spine in ax.spines.values():
            spine.set_color("#444466")
        ax.legend(fontsize=7, labelcolor="white", facecolor="#333344", edgecolor="#444466")

        # Value labels on bars
        for bar, val in zip(bars, vals):
            ypos = bar.get_height() + (abs(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=6.5)

    fig.suptitle("good-noise Benchmark Results — LibriSpeech test-clean\n5 speakers × 4 utterances × 6 attacks",
                 color="white", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "metrics_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Perturbation FFT detail for top 3 attacks
# ─────────────────────────────────────────────────────────────────────────────

def plot_perturbation_detail():
    clean_path = SAMPLES_DIR / REPR_SPEAKER / f"{REPR_UTTE}.wav"
    clean = load_wav(clean_path)
    n = min(len(clean), 4 * SR)
    clean = clean[:n]

    top_attacks = ["pgd_embedding", "psychoacoustic_pgd", "mel_disruption"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.patch.set_facecolor("#0F0F1A")
    freqs = np.fft.rfftfreq(n, 1 / SR)
    t = np.arange(n) / SR

    for j, aname in enumerate(top_attacks):
        out_path = OUTPUTS_DIR / f"{aname}_spk{REPR_SPEAKER}_{REPR_UTTE}.wav"
        protected = load_wav(out_path)[:n]
        delta = protected - clean
        color = COLORS[aname]

        # Time-domain perturbation
        ax_t = axes[0, j]
        ax_t.set_facecolor("#1A1A2E")
        ax_t.plot(t, delta, color=color, linewidth=0.5, alpha=0.8)
        ax_t.set_title(f"{ATTACK_LABELS[aname].replace(chr(10), ' ')}\nPerturbation (time)",
                       color="white", fontsize=8.5, fontweight="bold")
        ax_t.set_xlabel("Time (s)", color="#888899", fontsize=7)
        ax_t.set_ylabel("δ amplitude", color="#888899", fontsize=7)
        ax_t.tick_params(colors="#888899", labelsize=6)
        snr = 10 * np.log10(np.mean(clean**2) / (np.mean(delta**2) + 1e-12))
        ax_t.text(0.98, 0.97, f"SNR: {snr:.1f} dB", transform=ax_t.transAxes,
                  ha="right", va="top", color=color, fontsize=7.5, fontweight="bold")
        for spine in ax_t.spines.values():
            spine.set_color("#444466")

        # Frequency-domain perturbation vs clean
        ax_f = axes[1, j]
        ax_f.set_facecolor("#1A1A2E")
        clean_fft = np.abs(np.fft.rfft(clean)) / n
        delta_fft = np.abs(np.fft.rfft(delta)) / n
        ax_f.semilogy(freqs / 1000, clean_fft + 1e-8, color="#555566",
                      linewidth=0.7, alpha=0.8, label="Clean")
        ax_f.semilogy(freqs / 1000, delta_fft + 1e-8, color=color,
                      linewidth=0.8, alpha=0.9, label="Perturbation")
        ax_f.set_xlabel("Frequency (kHz)", color="#888899", fontsize=7)
        ax_f.set_ylabel("Magnitude", color="#888899", fontsize=7)
        ax_f.set_xlim(0, 8)
        ax_f.legend(fontsize=6.5, labelcolor="white", facecolor="#1A1A2E", edgecolor="#444466")
        ax_f.tick_params(colors="#888899", labelsize=6)
        for spine in ax_f.spines.values():
            spine.set_color("#444466")

    fig.suptitle("Perturbation Analysis: Top 3 Embedding Attacks\n(δ = protected − clean; imperceptible by design)",
                 color="white", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "perturbation_detail.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Speaker similarity by attack — dramatic results plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_speaker_sim_highlight():
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#1A1A2E")

    attacks = [r["attack"] for r in summary["results"]]
    sims = [r["mean_speaker_similarity"] for r in summary["results"]]
    colors = [COLORS[a] for a in attacks]
    labels = [ATTACK_LABELS[a].replace("\n", " ") for a in attacks]

    x = np.arange(len(attacks))
    bars = ax.barh(x, sims, height=0.55, color=colors, alpha=0.85, edgecolor="#333344")

    # Reference lines
    ax.axvline(1.0, color="#888899", linestyle="--", linewidth=1, alpha=0.5, label="Identity (no protection)")
    ax.axvline(0.25, color="#FF6B6B", linestyle="--", linewidth=1.5, alpha=0.9, label="Cloning failure threshold (<0.25)")
    ax.axvline(0.0, color="#888899", linestyle="-", linewidth=0.5, alpha=0.3)

    for bar, val, color in zip(bars, sims, colors):
        xpos = val + (0.03 if val >= 0 else -0.03)
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                ha=ha, va="center", color=color, fontsize=9, fontweight="bold")

    ax.set_yticks(x)
    ax.set_yticklabels(labels, color="white", fontsize=10)
    ax.set_xlabel("Speaker Cosine Similarity (after protection)", color="#888899", fontsize=9)
    ax.set_xlim(-1.1, 1.2)
    ax.tick_params(colors="#888899")
    for spine in ax.spines.values():
        spine.set_color("#444466")

    ax.legend(fontsize=9, labelcolor="white", facecolor="#1A1A2E", edgecolor="#444466", loc="lower right")
    ax.set_title(
        "Speaker Identity Disruption by Attack\n"
        "ECAPA-TDNN cosine similarity: 1.0 = same speaker, 0.25 = cloning fails, negative = opposite embedding",
        color="white", fontsize=10, fontweight="bold"
    )

    plt.tight_layout()
    out = RESULTS_DIR / "speaker_similarity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("\nGenerating visualizations...")
    plot_waveform_comparison()
    plot_spectrogram_grid()
    plot_metrics_bar()
    plot_perturbation_detail()
    plot_speaker_sim_highlight()
    print(f"\nAll figures saved to {RESULTS_DIR}/")
