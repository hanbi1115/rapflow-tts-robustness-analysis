import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

try:
    import whisper
except ImportError:
    whisper = None

try:
    from jiwer import wer
except ImportError:
    wer = None


def normalize_text(text):
    text = text.lower().strip()
    for ch in [".", ",", "!", "?", ";", ":", '"', "'", "(", ")", "[", "]"]:
        text = text.replace(ch, "")
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation_log",
        default="day8_stress_test/generation_log.csv"
    )
    parser.add_argument(
        "--output_dir",
        default="day9_stress_analysis"
    )
    parser.add_argument(
        "--whisper_model",
        default="tiny.en"
    )
    parser.add_argument(
        "--whisper_device",
        default="cpu"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.generation_log)
    if not log_path.exists():
        raise FileNotFoundError(f"Generation log not found: {log_path}")

    df = pd.read_csv(log_path)

    if whisper is None or wer is None:
        raise RuntimeError(
            "Missing packages. Run: pip install openai-whisper jiwer"
        )

    print(f"Loading Whisper model: {args.whisper_model} on {args.whisper_device}")
    asr = whisper.load_model(
        args.whisper_model,
        device=args.whisper_device
    )

    rows = []
    total = len(df)

    for i, row in df.iterrows():
        category = row["category"]
        item_id = int(row["item_id"])
        text = str(row["text"])
        nfe = int(row["nfe"])
        wav_path = Path(str(row["wav_path"]))

        print(
            f"[{i+1}/{total}] "
            f"{category} | item={item_id} | NFE={nfe}"
        )

        if not wav_path.exists():
            print(f"  missing wav: {wav_path}")
            rows.append({
                **row.to_dict(),
                "audio_duration_sec": np.nan,
                "rtf": np.nan,
                "transcript": "",
                "wer": np.nan,
                "eval_status": "missing_wav",
            })
            continue

        # Load at native sampling rate for duration
        y_native, sr_native = librosa.load(
            wav_path,
            sr=None,
            mono=True
        )
        duration = librosa.get_duration(
            y=y_native,
            sr=sr_native
        )

        # Load at 16 kHz and pass waveform directly to Whisper.
        # This avoids needing ffmpeg.
        y16, _ = librosa.load(
            wav_path,
            sr=16000,
            mono=True
        )
        y16 = y16.astype(np.float32)

        result = asr.transcribe(
            y16,
            language="en",
            fp16=False
        )

        transcript = normalize_text(result["text"])
        reference = normalize_text(text)
        current_wer = float(wer(reference, transcript))

        total_time = pd.to_numeric(
            pd.Series([row.get("total_time_sec")]),
            errors="coerce"
        ).iloc[0]

        rtf = (
            float(total_time) / float(duration)
            if pd.notna(total_time) and duration > 0
            else np.nan
        )

        print(f"  ref : {reference}")
        print(f"  hyp : {transcript}")
        print(f"  WER : {current_wer:.4f}")
        if not np.isnan(rtf):
            print(f"  RTF : {rtf:.4f}")

        rows.append({
            **row.to_dict(),
            "audio_duration_sec": duration,
            "rtf": rtf,
            "transcript": transcript,
            "wer": current_wer,
            "eval_status": "ok",
        })

    result_df = pd.DataFrame(rows)

    detailed_csv = out_dir / "detailed_results.csv"
    result_df.to_csv(
        detailed_csv,
        index=False,
        encoding="utf-8-sig"
    )

    valid = result_df[result_df["eval_status"] == "ok"].copy()

    # NFE-level summary
    by_nfe = (
        valid.groupby("nfe", as_index=False)
        .agg(
            mean_wer=("wer", "mean"),
            median_wer=("wer", "median"),
            mean_rtf=("rtf", "mean"),
            mean_total_time_sec=("total_time_sec", "mean"),
            mean_audio_duration_sec=("audio_duration_sec", "mean"),
            n_samples=("wer", "size"),
        )
    )
    by_nfe.to_csv(
        out_dir / "summary_by_nfe.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Category-level summary
    by_category = (
        valid.groupby("category", as_index=False)
        .agg(
            mean_wer=("wer", "mean"),
            median_wer=("wer", "median"),
            mean_rtf=("rtf", "mean"),
            n_samples=("wer", "size"),
        )
    )
    by_category.to_csv(
        out_dir / "summary_by_category.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Category x NFE summary: this is the main robustness table.
    category_nfe = (
        valid.groupby(["category", "nfe"], as_index=False)
        .agg(
            mean_wer=("wer", "mean"),
            median_wer=("wer", "median"),
            mean_rtf=("rtf", "mean"),
            mean_total_time_sec=("total_time_sec", "mean"),
            n_samples=("wer", "size"),
        )
    )
    category_nfe.to_csv(
        out_dir / "summary_category_by_nfe.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Pivot tables make patterns easy to inspect in Excel.
    wer_pivot = category_nfe.pivot(
        index="category",
        columns="nfe",
        values="mean_wer"
    )
    wer_pivot.to_csv(
        out_dir / "wer_pivot.csv",
        encoding="utf-8-sig"
    )

    rtf_pivot = category_nfe.pivot(
        index="category",
        columns="nfe",
        values="mean_rtf"
    )
    rtf_pivot.to_csv(
        out_dir / "rtf_pivot.csv",
        encoding="utf-8-sig"
    )

    # Find sentences where low-step and high-step behavior differ most.
    per_item = (
        valid.pivot_table(
            index=["category", "item_id", "text"],
            columns="nfe",
            values="wer",
            aggfunc="mean"
        )
        .reset_index()
    )

    if 1 in per_item.columns and 16 in per_item.columns:
        per_item["wer_gap_nfe1_minus_nfe16"] = (
            per_item[1] - per_item[16]
        )
        per_item = per_item.sort_values(
            "wer_gap_nfe1_minus_nfe16",
            ascending=False
        )

    per_item.to_csv(
        out_dir / "case_study_candidates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("Day 9 analysis complete.")
    print(f"Detailed results       : {detailed_csv}")
    print(f"By NFE                 : {out_dir / 'summary_by_nfe.csv'}")
    print(f"By category            : {out_dir / 'summary_by_category.csv'}")
    print(f"Category x NFE         : {out_dir / 'summary_category_by_nfe.csv'}")
    print(f"WER pivot              : {out_dir / 'wer_pivot.csv'}")
    print(f"RTF pivot              : {out_dir / 'rtf_pivot.csv'}")
    print(f"Case-study candidates  : {out_dir / 'case_study_candidates.csv'}")
    print("======================================")


if __name__ == "__main__":
    main()
