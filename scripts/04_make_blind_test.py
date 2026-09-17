import argparse
import csv
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root", default="day3_speed/audio")
    parser.add_argument("--output_dir", default="day6_blind")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nfe", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    args = parser.parse_args()

    audio_root = Path(args.audio_root)
    out_root = Path(args.output_dir)
    blind_audio_root = out_root / "audio"
    out_root.mkdir(parents=True, exist_ok=True)
    blind_audio_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    key_rows = []
    score_rows = []

    for sentence_dir in sorted(audio_root.glob("sentence_*")):
        sentence_id = int(sentence_dir.name.split("_")[-1])

        items = []
        for nfe in args.nfe:
            wav_path = sentence_dir / f"nfe_{nfe}.wav"
            if wav_path.exists():
                items.append((nfe, wav_path))

        rng.shuffle(items)

        letters = [chr(ord("A") + i) for i in range(len(items))]

        out_sentence_dir = blind_audio_root / f"sentence_{sentence_id}"
        out_sentence_dir.mkdir(parents=True, exist_ok=True)

        for letter, (nfe, src) in zip(letters, items):
            dst = out_sentence_dir / f"{letter}.wav"
            shutil.copy2(src, dst)

            key_rows.append({
                "sentence_id": sentence_id,
                "blind_label": letter,
                "nfe": nfe,
                "source_wav": str(src),
                "blind_wav": str(dst),
            })

            score_rows.append({
                "sentence_id": sentence_id,
                "blind_label": letter,
                "audio_file": str(dst),
                "naturalness_1to5": "",
                "clarity_1to5": "",
                "artifact_free_1to5": "",
                "overall_1to5": "",
                "notes": "",
            })

    key_path = out_root / "blind_key_DO_NOT_OPEN.csv"
    score_path = out_root / "blind_listening_scores.csv"

    with key_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=key_rows[0].keys())
        writer.writeheader()
        writer.writerows(key_rows)

    with score_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=score_rows[0].keys())
        writer.writeheader()
        writer.writerows(score_rows)

    instructions = out_root / "README_BLIND_TEST.txt"
    instructions.write_text(
        """Blind Listening Test

1. DO NOT open blind_key_DO_NOT_OPEN.csv before scoring.
2. Listen to each file in day6_blind/audio.
3. Fill blind_listening_scores.csv.

Scoring:
- naturalness_1to5: 1 = very unnatural, 5 = very natural
- clarity_1to5: 1 = unclear, 5 = very clear
- artifact_free_1to5: 1 = many glitches/noise, 5 = no obvious artifacts
- overall_1to5: your overall impression

Recommended:
- Use headphones.
- Keep volume fixed.
- Listen to each file at least twice.
- Do not compare filenames to the original experiment folders.
""",
        encoding="utf-8"
    )

    print("Blind listening package created.")
    print(f"Score sheet: {score_path}")
    print(f"Hidden key : {key_path}")
    print(f"Audio      : {blind_audio_root}")


if __name__ == "__main__":
    main()
