import argparse
import csv
import os
import time
from pathlib import Path

import torch
from scipy.io.wavfile import write

from src.utils import *
from text import text_to_sequence
from model import RapFlowTTS
from hifigan.denoiser import Denoiser


def synchronize(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def encode_text(text, cfg, model):
    x = text_to_sequence(text, cfg.preprocess.cleaner)[0]

    if cfg.model.add_blank:
        x = torch.tensor(
            intersperse(x, 0),
            dtype=torch.long,
            device=cfg.device
        )[None]
    else:
        x = torch.tensor(
            x,
            dtype=torch.long,
            device=cfg.device
        )[None]

    x_lengths = torch.tensor(
        [x.shape[-1]],
        dtype=torch.long,
        device=cfg.device
    )

    spk = (
        torch.tensor(
            [int(cfg.spk_id)],
            dtype=torch.long,
            device=cfg.device
        )
        if model.n_spks > 1
        else None
    )

    return x, x_lengths, spk


def synthesize_once(model, vocoder, denoiser, x, x_lengths, spk, cfg, nfe):
    # Keep the random seed fixed so NFE is the main changing factor.
    seed_init(seed=cfg.seed)

    synchronize(cfg.device)
    t0 = time.perf_counter()

    with torch.no_grad():
        output = model.synthesise(
            x,
            x_lengths,
            n_timesteps=nfe,
            temperature=cfg.temperature,
            spks=spk,
            length_scale=cfg.length_scale,
        )

    synchronize(cfg.device)
    t1 = time.perf_counter()

    with torch.no_grad():
        waveform = vocoder(output["mel"]).clamp(-1, 1)
        waveform = (
            denoiser(
                waveform.squeeze(0),
                strength=0.00025
            )
            .cpu()
            .squeeze()
            .numpy()
        )

    synchronize(cfg.device)
    t2 = time.perf_counter()

    return waveform, (t1 - t0), (t2 - t1), (t2 - t0)


def read_sentences(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "category": row["category"].strip(),
                "item_id": int(row["item_id"]),
                "text": row["text"].strip(),
                "purpose": row.get("purpose", "").strip(),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentences_csv",
        default="day7_stress_test_sentences.csv"
    )
    parser.add_argument(
        "--weight_path",
        default="./checkpoints/RapFlow-TTS-LJS-Stage3-Improved"
    )
    parser.add_argument("--weight_name", default="model-train-200")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spk_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.667)
    parser.add_argument("--length_scale", type=float, default=1.0)
    parser.add_argument(
        "--nfe",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16]
    )
    parser.add_argument(
        "--output_dir",
        default="day8_stress_test"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate wav files even if they already exist."
    )
    args = parser.parse_args()

    sentences = read_sentences(args.sentences_csv)
    total_jobs = len(sentences) * len(args.nfe)

    cfg_path = os.path.join(args.weight_path, "base.yaml")
    cfg = Config(cfg_path)
    cfg.device = args.device
    cfg.weight_path = args.weight_path
    cfg.weight_name = args.weight_name
    cfg.spk_id = args.spk_id
    cfg.seed = args.seed
    cfg.temperature = args.temperature
    cfg.length_scale = args.length_scale

    out_root = Path(args.output_dir)
    audio_root = out_root / "audio"
    out_root.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    print("Loading RapFlow-TTS once...")
    model = RapFlowTTS(cfg.model).to(cfg.device)

    ckpt = torch.load(
        os.path.join(cfg.weight_path, f"{cfg.weight_name}.pth"),
        map_location=cfg.device,
    )

    if cfg.test.ema:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["state_dict"], strict=False)

    model.eval()

    print("Loading HiFi-GAN once...")
    vocoder = get_vocoder(cfg, cfg.device)
    denoiser = Denoiser(vocoder, mode="zeros")

    # One warm-up run.
    print("GPU warm-up...")
    warm = sentences[0]
    x, x_lengths, spk = encode_text(warm["text"], cfg, model)
    _ = synthesize_once(
        model, vocoder, denoiser,
        x, x_lengths, spk, cfg,
        nfe=2
    )
    print("Warm-up complete.")

    results = []
    completed = 0

    for sentence in sentences:
        category = sentence["category"]
        item_id = sentence["item_id"]
        text = sentence["text"]
        purpose = sentence["purpose"]

        category_dir = audio_root / category
        category_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"[{category}] item {item_id}")
        print(text)

        x, x_lengths, spk = encode_text(text, cfg, model)

        for nfe in args.nfe:
            completed += 1
            wav_path = category_dir / f"item_{item_id:02d}_nfe_{nfe}.wav"

            if wav_path.exists() and not args.overwrite:
                print(
                    f"[{completed}/{total_jobs}] NFE={nfe}: "
                    f"already exists -> skip"
                )
                results.append({
                    "category": category,
                    "item_id": item_id,
                    "text": text,
                    "purpose": purpose,
                    "nfe": nfe,
                    "status": "skipped_existing",
                    "model_time_sec": "",
                    "vocoder_time_sec": "",
                    "total_time_sec": "",
                    "wav_path": str(wav_path),
                })
                continue

            try:
                waveform, model_time, vocoder_time, total_time = synthesize_once(
                    model,
                    vocoder,
                    denoiser,
                    x,
                    x_lengths,
                    spk,
                    cfg,
                    nfe,
                )

                write(wav_path, 22050, waveform)

                print(
                    f"[{completed}/{total_jobs}] NFE={nfe}: "
                    f"OK | model={model_time:.4f}s | "
                    f"total={total_time:.4f}s"
                )

                results.append({
                    "category": category,
                    "item_id": item_id,
                    "text": text,
                    "purpose": purpose,
                    "nfe": nfe,
                    "status": "ok",
                    "model_time_sec": round(model_time, 6),
                    "vocoder_time_sec": round(vocoder_time, 6),
                    "total_time_sec": round(total_time, 6),
                    "wav_path": str(wav_path),
                })

            except Exception as e:
                print(
                    f"[{completed}/{total_jobs}] NFE={nfe}: "
                    f"FAILED -> {repr(e)}"
                )

                results.append({
                    "category": category,
                    "item_id": item_id,
                    "text": text,
                    "purpose": purpose,
                    "nfe": nfe,
                    "status": f"failed: {repr(e)}",
                    "model_time_sec": "",
                    "vocoder_time_sec": "",
                    "total_time_sec": "",
                    "wav_path": "",
                })

            # Save progress after every single job, so a crash does not lose the log.
            progress_csv = out_root / "generation_log.csv"
            with progress_csv.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

    print("\n" + "=" * 70)
    print("Day 8 generation complete.")
    print(f"Total planned jobs : {total_jobs}")
    print(f"Audio folder       : {audio_root}")
    print(f"Generation log     : {out_root / 'generation_log.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
