import argparse
import csv
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

try:
    import whisper
except ImportError:
    whisper = None

try:
    from jiwer import wer
except ImportError:
    wer = None


REFERENCE_SENTENCES = {
    1: "This is my first RapFlow TTS experiment.",
    2: "The weather is surprisingly nice today.",
    3: "Although the experiment was difficult, the final result was better than we expected.",
}


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def safe_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.std(values)) if len(values) else float("nan")


def analyze_audio(wav_path):
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y).squeeze()

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).squeeze()

    return {
        "sample_rate": sr,
        "duration_sec": duration,
        "mean_rms": safe_mean(rms),
        "std_rms": safe_std(rms),
        "mean_f0_hz": safe_mean(f0),
        "std_f0_hz": safe_std(f0),
        "voiced_ratio": (
            float(np.mean(voiced_flag.astype(float)))
            if voiced_flag is not None and len(voiced_flag) > 0
            else float("nan")
        ),
        "mean_spectral_centroid_hz": safe_mean(centroid),
    }


def save_spectrogram(wav_path, out_path):
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=80,
        fmax=sr / 2,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        fmax=sr / 2,
    )
    plt.title(wav_path.stem)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def normalize_text(text):
    text = text.lower().strip()
    for ch in [".", ",", "!", "?", ";", ":", '"', "'"]:
        text = text.replace(ch, "")
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root", default="day3_speed/audio")
    parser.add_argument("--output_dir", default="day4_quality")
    parser.add_argument("--nfe", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--whisper_model", default="tiny.en")
    parser.add_argument("--whisper_device", default="cpu")
    parser.add_argument("--skip_whisper", action="store_true")
    args = parser.parse_args()

    audio_root = Path(args.audio_root)
    out_root = Path(args.output_dir)
    spec_root = out_root / "spectrograms"
    out_root.mkdir(parents=True, exist_ok=True)
    spec_root.mkdir(parents=True, exist_ok=True)

    if not audio_root.exists():
        raise FileNotFoundError(f"Audio folder not found: {audio_root}")

    use_whisper = not args.skip_whisper

    if use_whisper:
        if whisper is None or wer is None:
            raise RuntimeError("Run: pip install openai-whisper jiwer")
        print(f"Loading Whisper model: {args.whisper_model} on {args.whisper_device}")
        asr_model = whisper.load_model(args.whisper_model, device=args.whisper_device)
    else:
        asr_model = None

    rows = []

    for sentence_id, reference in REFERENCE_SENTENCES.items():
        sentence_dir = audio_root / f"sentence_{sentence_id}"

        for nfe in args.nfe:
            wav_path = sentence_dir / f"nfe_{nfe}.wav"
            if not wav_path.exists():
                print(f"[missing] {wav_path}")
                continue

            print(f"\nSentence {sentence_id} | NFE={nfe}")

            metrics = analyze_audio(wav_path)
            transcript = ""
            current_wer = float("nan")

            if use_whisper:
                # Load waveform ourselves at Whisper's required 16 kHz.
                # Passing a NumPy waveform avoids Whisper's ffmpeg subprocess entirely.
                y16, _ = librosa.load(wav_path, sr=16000, mono=True)
                y16 = y16.astype(np.float32)

                result = asr_model.transcribe(
                    y16,
                    language="en",
                    fp16=False,
                )
                transcript = normalize_text(result["text"])
                ref_norm = normalize_text(reference)
                current_wer = float(wer(ref_norm, transcript))

                print(f"Reference : {ref_norm}")
                print(f"Whisper   : {transcript}")
                print(f"WER       : {current_wer:.4f}")

            spec_dir = spec_root / f"sentence_{sentence_id}"
            spec_dir.mkdir(parents=True, exist_ok=True)
            spec_path = spec_dir / f"nfe_{nfe}.png"
            save_spectrogram(wav_path, spec_path)

            rows.append({
                "sentence_id": sentence_id,
                "reference": reference,
                "nfe": nfe,
                "wav_path": str(wav_path),
                "transcript": transcript,
                "wer": current_wer,
                **metrics,
                "spectrogram_path": str(spec_path),
            })

    if not rows:
        raise RuntimeError("No wav files were found.")

    csv_path = out_root / "quality_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nDay 4 complete.")
    print(f"Metrics CSV  : {csv_path}")
    print(f"Spectrograms : {spec_root}")


if __name__ == "__main__":
    main()
