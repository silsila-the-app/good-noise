"""Full benchmark: run all 6 attacks across 5 speakers (4 utterances each).

Logs every result to experiments/ledger.jsonl and produces a summary table.
"""

import sys, os, json, datetime, warnings, hashlib
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config ───────────────────────────────────────────────────────────────────
CFG_PATH = "config/default.yaml"
SAMPLES_DIR = Path("samples")
OUTPUT_DIR = Path("outputs/benchmark")
LEDGER_PATH = Path("experiments/ledger.jsonl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = yaml.safe_load(open(CFG_PATH))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACKS_TO_RUN = [
    "spectral_filter",
    "mel_disruption",
    "pgd_embedding",
    "psychoacoustic_pgd",
    "universal_pgd",
    "asr_disruption",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_wav(path: str) -> tuple:
    wav_np, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    peak = np.abs(wav_np).max()
    if peak > 1e-6:
        wav_np = wav_np / peak
    return torch.from_numpy(wav_np).unsqueeze(0).to(DEVICE), sr


def get_git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def log_entry(entry: dict):
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Collect sample files ──────────────────────────────────────────────────────
speaker_dirs = sorted([d for d in SAMPLES_DIR.iterdir() if d.is_dir()])
all_files = []
for spk_dir in speaker_dirs[:5]:
    wavs = sorted(spk_dir.glob("*.wav"))[:4]
    for wav_path in wavs:
        all_files.append((spk_dir.name, str(wav_path)))

print(f"\n{'='*65}")
print(f"  good-noise benchmark | {len(all_files)} files | device={DEVICE}")
print(f"{'='*65}")
print(f"  Attacks: {ATTACKS_TO_RUN}")
print(f"{'='*65}\n")

# ── Import attacks and metrics ────────────────────────────────────────────────
from attacks import ATTACK_REGISTRY
from metrics.audio_metrics import compute_all_metrics

# Pre-load shared models once to avoid re-downloading each run
print("Pre-loading models...")
# Load speaker encoder once
from models.speaker_encoder import SpeakerEncoder
_shared_encoder = SpeakerEncoder(DEVICE)
dummy_wav = torch.zeros(1, 16000, device=DEVICE)
_ = _shared_encoder.embed(dummy_wav)
print("  ECAPA-TDNN encoder ready")

# ── Build attack instances ────────────────────────────────────────────────────
attack_instances = {}
for aname in ATTACKS_TO_RUN:
    attack_instances[aname] = ATTACK_REGISTRY[aname](cfg, DEVICE)
    # Inject shared encoder for encoder-based attacks
    if hasattr(attack_instances[aname], "encoder"):
        attack_instances[aname].encoder = _shared_encoder

# Universal PGD: train on first speaker's corpus
print("\nTraining universal perturbation on speaker 6930 corpus...")
upd_atk = attack_instances["universal_pgd"]
corpus_dir = SAMPLES_DIR / speaker_dirs[0].name
corpus_wavs = [load_wav(str(p))[0] for p in sorted(corpus_dir.glob("*.wav"))]
upd_atk.train(corpus_wavs)
print("  Universal delta trained.")

# ── Benchmark loop ────────────────────────────────────────────────────────────
results_summary = {aname: [] for aname in ATTACKS_TO_RUN}
git_commit = get_git_commit()

for spk_id, wav_path in all_files:
    print(f"\n  Speaker {spk_id} | {Path(wav_path).name}")
    wav, sr = load_wav(wav_path)

    for aname in ATTACKS_TO_RUN:
        atk = attack_instances[aname]

        protected = atk.apply(wav.clone(), sr)

        # Save output
        out_name = f"{aname}_spk{spk_id}_{Path(wav_path).stem}.wav"
        out_path = OUTPUT_DIR / out_name
        wav_out = protected.squeeze().cpu().float().numpy()
        sf.write(str(out_path), wav_out, sr)

        # Metrics
        metrics = compute_all_metrics(
            wav.cpu(), protected.cpu(), sr, DEVICE, cfg
        )
        metrics_clean = {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in metrics.items() if v is not None}
        results_summary[aname].append(metrics_clean)

        snr = metrics.get("snr", 0) or 0
        pesq = metrics.get("pesq") or 0
        stoi = metrics.get("stoi") or 0
        sim = metrics.get("speaker_similarity") or 0
        print(f"    [{aname:22s}] SNR={snr:5.1f}dB PESQ={pesq:.2f} STOI={stoi:.3f} SpeakerSim={sim:.3f}")

        # Log
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "id": f"{aname}_{spk_id}_{Path(wav_path).stem}",
            "attack": aname,
            "speaker_id": spk_id,
            "input_file": wav_path,
            "input_hash": file_hash(wav_path),
            "output_file": str(out_path),
            "git_commit": git_commit,
            "config_path": CFG_PATH,
            "sample_rate": sr,
            "duration_s": round(wav.shape[-1] / sr, 3),
            "metrics": metrics_clean,
            "attack_cfg": cfg.get("attacks", {}).get(aname, {}),
            "status": "completed",
        }
        log_entry(entry)

# ── Aggregate summary ─────────────────────────────────────────────────────────
print(f"\n\n{'='*65}")
print("  BENCHMARK SUMMARY (mean across all speakers x utterances)")
print(f"{'='*65}")
print(f"  {'Attack':<22} {'SNR(dB)':>8} {'PESQ':>7} {'STOI':>7} {'SpeakerSim':>12}")
print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*7} {'-'*12}")

summary_rows = []
for aname in ATTACKS_TO_RUN:
    rows = results_summary[aname]
    def mean(key):
        vals = [r[key] for r in rows if key in r]
        return float(np.mean(vals)) if vals else float("nan")

    snr = mean("snr")
    pesq = mean("pesq")
    stoi = mean("stoi")
    sim = mean("speaker_similarity")
    print(f"  {aname:<22} {snr:>8.2f} {pesq:>7.3f} {stoi:>7.3f} {sim:>12.3f}")
    summary_rows.append({
        "attack": aname,
        "mean_snr": round(snr, 3),
        "mean_pesq": round(pesq, 3),
        "mean_stoi": round(stoi, 3),
        "mean_speaker_similarity": round(sim, 3),
    })

print(f"{'='*65}")
print("\nTargets: SNR>20dB | PESQ>3.5 | STOI>0.90 | SpeakerSim<0.25\n")

# Save summary JSON
summary = {
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "git_commit": git_commit,
    "n_speakers": len(speaker_dirs[:5]),
    "n_utterances_per_speaker": 4,
    "device": str(DEVICE),
    "results": summary_rows,
}
with open("experiments/benchmark_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Summary saved to experiments/benchmark_summary.json")
print(f"All {len(all_files) * len(ATTACKS_TO_RUN)} runs logged to {LEDGER_PATH}")
