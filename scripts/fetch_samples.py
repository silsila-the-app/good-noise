"""Download LibriSpeech test-clean samples via HuggingFace streaming.

Saves WAV files to samples/<speaker_id>/<utterance_id>.wav
Selects N_SPEAKERS speakers with N_UTTES utterances each.
"""

import sys
import os
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLES_DIR = Path("samples")
N_SPEAKERS = 5
N_UTTES = 4  # per speaker
TARGET_SR = 16000


def fetch_samples():
    from datasets import load_dataset

    print("Streaming LibriSpeech test-clean (no full download needed)...")
    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        streaming=True,
    )

    speakers_seen = {}

    for sample in ds:
        sid = str(sample["speaker_id"])
        if sid not in speakers_seen:
            speakers_seen[sid] = []
        if len(speakers_seen[sid]) >= N_UTTES:
            continue

        audio = sample["audio"]
        wav = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        # Resample to 16 kHz if needed
        if sr != TARGET_SR:
            import torchaudio, torch
            t = torch.from_numpy(wav).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, TARGET_SR)
            wav = t.squeeze(0).numpy()

        # Normalize
        peak = np.abs(wav).max()
        if peak > 1e-6:
            wav = wav / peak

        # Skip very short clips
        if len(wav) < TARGET_SR:
            continue

        out_dir = SAMPLES_DIR / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = len(speakers_seen[sid])
        out_path = out_dir / f"utte{idx:02d}.wav"
        sf.write(str(out_path), wav, TARGET_SR)
        speakers_seen[sid].append(str(out_path))
        print(f"  Saved {out_path} ({len(wav)/TARGET_SR:.1f}s)")

        if all(len(v) >= N_UTTES for v in speakers_seen.values()) and len(speakers_seen) >= N_SPEAKERS:
            break

    print(f"\nDone. {len(speakers_seen)} speakers, paths saved.")
    return speakers_seen


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    speakers = fetch_samples()
    print("\nSpeaker → utterance paths:")
    for sid, paths in speakers.items():
        print(f"  {sid}: {paths}")
