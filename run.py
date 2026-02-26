#!/usr/bin/env python3
"""good-noise — canonical entry point.

Applies adversarial audio perturbation to protect voice recordings from
unauthorized AI training and voice cloning.

Usage:
    python run.py input.wav [options]

Examples:
    # Apply default attack (PGD embedding) with default config
    python run.py podcast.wav

    # Apply specific attack
    python run.py voice.wav --attack psychoacoustic_pgd

    # Apply multiple attacks (each saves a separate output)
    python run.py voice.wav --attack pgd_embedding mel_disruption

    # Use custom config
    python run.py voice.wav --config config/default.yaml --attack spectral_filter

    # Train universal perturbation on a folder, then apply
    python run.py voice.wav --attack universal_pgd --corpus samples/

    # Save output to specific path
    python run.py voice.wav --output protected/voice_protected.wav
"""

import argparse
import json
import hashlib
import datetime
import sys
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import yaml
import numpy as np


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_audio(path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    """Load audio, resample to target_sr, peak-normalize."""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Peak normalize
    peak = waveform.abs().max()
    if peak > 1e-6:
        waveform = waveform / peak
    return waveform, target_sr


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wav_np = waveform.squeeze().cpu().float().numpy()
    sf.write(path, wav_np, sample_rate)


def log_experiment(ledger_path: str, entry: dict):
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(
        description="good-noise: adversarial audio protection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input audio file (.wav, .flac, .mp3)")
    parser.add_argument(
        "--attack",
        nargs="+",
        default=None,
        help="Attack(s) to apply. Options: pgd_embedding, psychoacoustic_pgd, "
             "mel_disruption, universal_pgd, asr_disruption, spectral_filter. "
             "Defaults to config value.",
    )
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path")
    parser.add_argument("--output", default=None, help="Output path (default: outputs/<attack>_<input>)")
    parser.add_argument("--corpus", default=None, help="Folder of .wav files for universal_pgd training")
    parser.add_argument("--device", default=None, help="cuda / cpu / mps (auto-detect if omitted)")
    parser.add_argument("--no-metrics", action="store_true", help="Skip metric computation")
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"good-noise | device={device} | config={args.config}")

    # Determine which attacks to run
    attack_names = args.attack or cfg.get("attacks", {}).get("active", ["pgd_embedding"])

    # Load input audio
    sample_rate = cfg.get("audio", {}).get("sample_rate", 16000)
    waveform, sample_rate = load_audio(args.input, sample_rate)
    waveform = waveform.to(device)
    print(f"Input: {args.input} | {waveform.shape[-1]/sample_rate:.2f}s @ {sample_rate} Hz")

    # ── Attack loop ──────────────────────────────────────────────────────────
    from attacks import ATTACK_REGISTRY
    from metrics.audio_metrics import compute_all_metrics, print_metrics

    output_dir = cfg.get("logging", {}).get("output_dir", "outputs/")
    ledger_path = cfg.get("logging", {}).get("ledger", "experiments/ledger.jsonl")
    input_stem = Path(args.input).stem

    for attack_name in attack_names:
        if attack_name not in ATTACK_REGISTRY:
            print(f"[WARN] Unknown attack '{attack_name}'. Available: {list(ATTACK_REGISTRY.keys())}")
            continue

        print(f"\n{'═'*55}")
        print(f"  Running attack: {attack_name}")
        print(f"{'═'*55}")

        attack = ATTACK_REGISTRY[attack_name](cfg, device)

        # Special handling for universal_pgd: load corpus
        if attack_name == "universal_pgd" and args.corpus:
            corpus_paths = sorted(Path(args.corpus).glob("*.wav"))[:cfg.get("attacks", {}).get("universal_pgd", {}).get("n_samples", 10)]
            corpus = [load_audio(str(p), sample_rate)[0].to(device) for p in corpus_paths]
            print(f"  Training universal delta on {len(corpus)} utterances from {args.corpus}")
            attack.train(corpus)

        # Apply attack
        protected = attack.apply(waveform, sample_rate)

        # Determine output path
        if args.output and len(attack_names) == 1:
            out_path = args.output
        else:
            out_path = str(Path(output_dir) / f"{attack_name}_{input_stem}.wav")

        save_audio(protected, out_path, sample_rate)
        print(f"  Saved: {out_path}")

        # ── Metrics ──────────────────────────────────────────────────────────
        metrics = {}
        if not args.no_metrics:
            print(f"  Computing metrics...")
            metrics = compute_all_metrics(
                waveform.cpu(), protected.cpu(), sample_rate, device, cfg
            )
            print_metrics(metrics, attack_name)

        # ── Log experiment ────────────────────────────────────────────────────
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "id": f"{attack_name}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "attack": attack_name,
            "input_file": args.input,
            "input_hash": file_hash(args.input),
            "output_file": out_path,
            "git_commit": get_git_commit(),
            "config_path": args.config,
            "sample_rate": sample_rate,
            "duration_s": round(waveform.shape[-1] / sample_rate, 3),
            "metrics": metrics,
            "attack_cfg": cfg.get("attacks", {}).get(attack_name, {}),
            "status": "completed",
        }
        log_experiment(ledger_path, entry)

    print(f"\nDone. Experiment(s) logged to {ledger_path}")


if __name__ == "__main__":
    main()
