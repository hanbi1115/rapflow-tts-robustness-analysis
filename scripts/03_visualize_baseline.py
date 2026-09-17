import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed_csv", default="day3_speed/speed_summary.csv")
    parser.add_argument("--quality_csv", default="day4_quality/quality_metrics.csv")
    parser.add_argument("--output_dir", default="day5_analysis")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speed = pd.read_csv(args.speed_csv)
    quality = pd.read_csv(args.quality_csv)

    # Aggregate by NFE
    speed_agg = (
        speed.groupby("nfe", as_index=False)
        .agg(
            mean_model_time_sec=("mean_model_time_sec", "mean"),
            mean_total_time_sec=("mean_total_time_sec", "mean"),
            mean_rtf=("mean_rtf", "mean"),
        )
    )

    quality_agg = (
        quality.groupby("nfe", as_index=False)
        .agg(
            mean_wer=("wer", "mean"),
            mean_f0_hz=("mean_f0_hz", "mean"),
            mean_rms=("mean_rms", "mean"),
            mean_voiced_ratio=("voiced_ratio", "mean"),
            mean_spectral_centroid_hz=("mean_spectral_centroid_hz", "mean"),
        )
    )

    merged = pd.merge(speed_agg, quality_agg, on="nfe", how="inner")
    merged.to_csv(out_dir / "merged_summary.csv", index=False, encoding="utf-8-sig")

    # 1) NFE vs model inference time
    plt.figure(figsize=(7, 5))
    plt.plot(merged["nfe"], merged["mean_model_time_sec"], marker="o")
    plt.xlabel("NFE / sampling steps")
    plt.ylabel("Mean model inference time (s)")
    plt.title("NFE vs Model Inference Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "nfe_vs_model_time.png", dpi=180)
    plt.close()

    # 2) NFE vs RTF
    plt.figure(figsize=(7, 5))
    plt.plot(merged["nfe"], merged["mean_rtf"], marker="o")
    plt.xlabel("NFE / sampling steps")
    plt.ylabel("Mean RTF")
    plt.title("NFE vs Real-Time Factor")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "nfe_vs_rtf.png", dpi=180)
    plt.close()

    # 3) NFE vs WER
    plt.figure(figsize=(7, 5))
    plt.plot(merged["nfe"], merged["mean_wer"], marker="o")
    plt.xlabel("NFE / sampling steps")
    plt.ylabel("Mean WER")
    plt.title("NFE vs ASR Word Error Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "nfe_vs_wer.png", dpi=180)
    plt.close()

    # 4) NFE vs spectral centroid
    plt.figure(figsize=(7, 5))
    plt.plot(merged["nfe"], merged["mean_spectral_centroid_hz"], marker="o")
    plt.xlabel("NFE / sampling steps")
    plt.ylabel("Mean spectral centroid (Hz)")
    plt.title("NFE vs Spectral Centroid")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "nfe_vs_spectral_centroid.png", dpi=180)
    plt.close()

    # 5) Speed-quality trade-off: RTF vs spectral centroid
    plt.figure(figsize=(7, 5))
    plt.scatter(
        merged["mean_rtf"],
        merged["mean_spectral_centroid_hz"],
        s=70,
    )
    for _, row in merged.iterrows():
        plt.annotate(
            f'NFE={int(row["nfe"])}',
            (row["mean_rtf"], row["mean_spectral_centroid_hz"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    plt.xlabel("Mean RTF (lower is faster)")
    plt.ylabel("Mean spectral centroid (Hz)")
    plt.title("Speed–Acoustic Trade-off")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "speed_quality_tradeoff.png", dpi=180)
    plt.close()

    # Manual listening sheet template
    listening = quality[["sentence_id", "nfe", "wav_path"]].copy()
    listening["naturalness_1to5"] = ""
    listening["clarity_1to5"] = ""
    listening["artifacts_1to5"] = ""
    listening["notes"] = ""
    listening.to_csv(
        out_dir / "listening_scores_template.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Day 5 analysis complete.")
    print(f"Output folder: {out_dir}")
    print("Created:")
    for p in sorted(out_dir.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
