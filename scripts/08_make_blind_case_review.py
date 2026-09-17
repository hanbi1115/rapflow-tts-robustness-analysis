import argparse
import csv
import random
import shutil
from pathlib import Path

import pandas as pd


NFE_ORDER = [1, 2, 4, 8, 16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected_cases",
        default="day10_visualization/selected_case_studies.csv"
    )
    parser.add_argument(
        "--audio_root",
        default="day8_stress_test/audio"
    )
    parser.add_argument(
        "--output_dir",
        default="day11_blind_case_review"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260917
    )
    args = parser.parse_args()

    cases = pd.read_csv(args.selected_cases)
    audio_root = Path(args.audio_root)
    out_root = Path(args.output_dir)
    audio_out = out_root / "audio"

    out_root.mkdir(parents=True, exist_ok=True)
    audio_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    key_rows = []
    score_rows = []

    for case_idx, row in cases.iterrows():
        case_id = case_idx + 1
        category = str(row["category"])
        item_id = int(row["item_id"])
        text = str(row["text"])

        case_dir = audio_out / f"case_{case_id:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        items = []
        for nfe in NFE_ORDER:
            src = audio_root / category / f"item_{item_id:02d}_nfe_{nfe}.wav"

            if not src.exists():
                raise FileNotFoundError(
                    f"Missing audio file: {src}"
                )

            items.append((nfe, src))

        rng.shuffle(items)

        labels = [chr(ord("A") + i) for i in range(len(items))]

        for label, (nfe, src) in zip(labels, items):
            dst = case_dir / f"{label}.wav"
            shutil.copy2(src, dst)

            key_rows.append({
                "case_id": case_id,
                "category": category,
                "item_id": item_id,
                "text": text,
                "blind_label": label,
                "nfe": nfe,
                "source_wav": str(src),
                "blind_wav": str(dst),
            })

            score_rows.append({
                "case_id": case_id,
                "category": category,
                "text": text,
                "blind_label": label,
                "audio_file": str(dst),
                "word_content_1to5": "",
                "pronunciation_1to5": "",
                "naturalness_1to5": "",
                "obvious_error_yes_no": "",
                "heard_text_or_error": "",
                "notes": "",
            })

    key_path = out_root / "blind_key_DO_NOT_OPEN.csv"
    score_path = out_root / "blind_case_scores.csv"

    with key_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=key_rows[0].keys())
        writer.writeheader()
        writer.writerows(key_rows)

    with score_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=score_rows[0].keys())
        writer.writeheader()
        writer.writerows(score_rows)

    instructions = out_root / "README_DAY11.txt"
    instructions.write_text(
        """DAY 11: Blind Case Review

Goal:
Check whether the NFE-sensitive cases found by Whisper are real TTS differences
or just ASR recognition errors.

IMPORTANT:
Do NOT open blind_key_DO_NOT_OPEN.csv until scoring is complete.

For each audio file, fill:

1) word_content_1to5
   5 = all intended words/content are clearly present
   4 = tiny uncertainty, but content is basically correct
   3 = one noticeable omission/repetition/substitution
   2 = several content errors
   1 = major content failure

2) pronunciation_1to5
   5 = clearly pronounced
   4 = slightly awkward but understandable
   3 = noticeable pronunciation problem
   2 = hard to understand
   1 = very poor / unintelligible

3) naturalness_1to5
   5 = very natural
   4 = mostly natural
   3 = somewhat synthetic or awkward
   2 = clearly unnatural
   1 = severely unnatural

4) obvious_error_yes_no
   Write Y if you hear an obvious omission, repetition, wrong word,
   severe pronunciation problem, or glitch.
   Otherwise write N.

5) heard_text_or_error
   If something sounds wrong, write what you actually heard or describe the error.
   Example:
   - "go" repeated 7 times
   - "seashells" sounded like "sea cells"
   - skipped "yellow"

Recommended procedure:
- Use headphones.
- Keep volume fixed.
- Listen to every file at least twice.
- For each case, score A-E without trying to guess which NFE it is.
- Do not compare against the original NFE filenames.

After scoring:
Upload both blind_case_scores.csv and blind_key_DO_NOT_OPEN.csv.
""",
        encoding="utf-8"
    )

    print("Day 11 blind case-review package created.")
    print(f"Audio folder : {audio_out}")
    print(f"Score sheet  : {score_path}")
    print(f"Hidden key   : {key_path}")
    print(f"Instructions : {instructions}")
    print(f"Total files  : {len(score_rows)}")


if __name__ == "__main__":
    main()
