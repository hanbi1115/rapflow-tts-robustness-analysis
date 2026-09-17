import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import torch
from scipy.io.wavfile import write

from src.utils import *
from text import text_to_sequence
from model import RapFlowTTS
from hifigan.denoiser import Denoiser


DEFAULT_SENTENCES = [
    "This is my first RapFlow TTS experiment.",
    "The weather is surprisingly nice today.",
    "Although the experiment was difficult, the final result was better than we expected."
]


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

    model_time = t1 - t0
    vocoder_time = t2 - t1
    total_time = t2 - t0
    duration = len(waveform) / 22050.0
    rtf = total_time / duration if duration > 0 else float("nan")

    return waveform, model_time, vocoder_time, total_time, duration, rtf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--weight_path",
        default="./checkpoints/RapFlow-TTS-LJS-Stage3-Improved"
    )
    parser.add_argument("--weight_name", default="model-train-200")
    parser.add_argument("--model_name", default="RapFlowTTS")
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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output_dir", default="day3_speed")
    args = parser.parse_args()

    cfg_path = os.path.join(args.weight_path, "base.yaml")
    cfg = Config(cfg_path)

    cfg.device = args.device
    cfg.weight_path = args.weight_path
    cfg.weight_name = args.weight_name
    cfg.model_name = args.model_name
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

    print("GPU warm-up...")
    warm_x, warm_len, warm_spk = encode_text(DEFAULT_SENTENCES[0], cfg, model)
    _ = synthesize_once(
        model,
        vocoder,
        denoiser,
        warm_x,
        warm_len,
        warm_spk,
        cfg,
        nfe=2,
    )
    print("Warm-up complete.")

    raw_rows = []
    summary_rows = []

    for sentence_id, sentence in enumerate(DEFAULT_SENTENCES, start=1):
        print(f"\n===== Sentence {sentence_id} =====")
        print(sentence)

        x, x_lengths, spk = encode_text(sentence, cfg, model)

        for nfe in args.nfe:
            total_times = []
            model_times = []
            vocoder_times = []
            durations = []
            rtfs = []
            saved_waveform = None

            print(f"\nNFE={nfe}")

            for rep in range(1, args.repeats + 1):
                waveform, model_time, vocoder_time, total_time, duration, rtf = synthesize_once(
                    model,
                    vocoder,
                    denoiser,
                    x,
                    x_lengths,
                    spk,
                    cfg,
                    nfe,
                )

                saved_waveform = waveform
                model_times.append(model_time)
                vocoder_times.append(vocoder_time)
                total_times.append(total_time)
                durations.append(duration)
                rtfs.append(rtf)

                print(
                    f"  repeat {rep}: "
                    f"model={model_time:.4f}s | "
                    f"vocoder={vocoder_time:.4f}s | "
                    f"total={total_time:.4f}s | "
                    f"duration={duration:.3f}s | "
                    f"RTF={rtf:.4f}"
                )

                raw_rows.append({
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "nfe": nfe,
                    "repeat": rep,
                    "model_time_sec": round(model_time, 6),
                    "vocoder_time_sec": round(vocoder_time, 6),
                    "total_time_sec": round(total_time, 6),
                    "audio_duration_sec": round(duration, 6),
                    "rtf": round(rtf, 6),
                })

            sentence_dir = audio_root / f"sentence_{sentence_id}"
            sentence_dir.mkdir(parents=True, exist_ok=True)
            wav_path = sentence_dir / f"nfe_{nfe}.wav"
            write(wav_path, 22050, saved_waveform)

            summary_rows.append({
                "sentence_id": sentence_id,
                "sentence": sentence,
                "nfe": nfe,
                "repeats": args.repeats,
                "mean_model_time_sec": round(statistics.mean(model_times), 6),
                "mean_vocoder_time_sec": round(statistics.mean(vocoder_times), 6),
                "mean_total_time_sec": round(statistics.mean(total_times), 6),
                "median_total_time_sec": round(statistics.median(total_times), 6),
                "mean_audio_duration_sec": round(statistics.mean(durations), 6),
                "mean_rtf": round(statistics.mean(rtfs), 6),
                "median_rtf": round(statistics.median(rtfs), 6),
                "wav_path": str(wav_path),
            })

    raw_csv = out_root / "raw_timings.csv"
    summary_csv = out_root / "speed_summary.csv"

    with raw_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
        writer.writeheader()
        writer.writerows(raw_rows)

    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=================================")
    print("Day 3 benchmark complete.")
    print(f"Raw timings : {raw_csv}")
    print(f"Summary     : {summary_csv}")
    print(f"Audio files : {audio_root}")
    print("=================================")


if __name__ == "__main__":
    main()
