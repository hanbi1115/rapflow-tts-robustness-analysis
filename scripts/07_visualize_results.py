import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NFE_ORDER = [1, 2, 4, 8, 16]


def save_line_plot(x, y, xlabel, ylabel, title, out_path, xticks=None):
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    if xticks is not None:
        plt.xticks(xticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_by_nfe",
        default="day9_stress_analysis/summary_by_nfe.csv"
    )
    parser.add_argument(
        "--summary_by_category",
        default="day9_stress_analysis/summary_by_category.csv"
    )
    parser.add_argument(
        "--summary_category_by_nfe",
        default="day9_stress_analysis/summary_category_by_nfe.csv"
    )
    parser.add_argument(
        "--case_study_candidates",
        default="day9_stress_analysis/case_study_candidates.csv"
    )
    parser.add_argument(
        "--output_dir",
        default="day10_visualization"
    )
    parser.add_argument(
        "--top_cases",
        type=int,
        default=5
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    case_dir = out_dir / "case_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    by_nfe = pd.read_csv(args.summary_by_nfe)
    by_category = pd.read_csv(args.summary_by_category)
    cat_nfe = pd.read_csv(args.summary_category_by_nfe)
    cases = pd.read_csv(args.case_study_candidates)

    # ------------------------------------------------------------
    # 1) Overall NFE vs WER
    # ------------------------------------------------------------
    by_nfe = by_nfe.sort_values("nfe")
    save_line_plot(
        by_nfe["nfe"],
        by_nfe["mean_wer"],
        "NFE / sampling steps",
        "Mean WER",
        "Overall NFE vs Word Error Rate",
        out_dir / "overall_nfe_vs_wer.png",
        xticks=NFE_ORDER,
    )

    # ------------------------------------------------------------
    # 2) Overall NFE vs RTF
    # ------------------------------------------------------------
    save_line_plot(
        by_nfe["nfe"],
        by_nfe["mean_rtf"],
        "NFE / sampling steps",
        "Mean RTF",
        "Overall NFE vs Real-Time Factor",
        out_dir / "overall_nfe_vs_rtf.png",
        xticks=NFE_ORDER,
    )

    # ------------------------------------------------------------
    # 3) Mean WER by category
    # ------------------------------------------------------------
    cat_sorted = by_category.sort_values("mean_wer", ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(cat_sorted["category"], cat_sorted["mean_wer"])
    plt.xlabel("Input category")
    plt.ylabel("Mean WER")
    plt.title("Mean WER by Input Category")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_wer_by_category.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # 4) Category x NFE WER heatmap
    # ------------------------------------------------------------
    pivot = cat_nfe.pivot(
        index="category",
        columns="nfe",
        values="mean_wer"
    )

    pivot = pivot.reindex(columns=NFE_ORDER)
    pivot = pivot.sort_index()

    matrix = pivot.to_numpy()

    plt.figure(figsize=(8, 6))
    image = plt.imshow(matrix, aspect="auto")
    plt.colorbar(image, label="Mean WER")
    plt.xticks(
        range(len(pivot.columns)),
        [str(x) for x in pivot.columns]
    )
    plt.yticks(
        range(len(pivot.index)),
        pivot.index
    )
    plt.xlabel("NFE / sampling steps")
    plt.ylabel("Input category")
    plt.title("WER Across Categories and NFE")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                plt.text(
                    j, i,
                    f"{value:.2f}",
                    ha="center",
                    va="center"
                )

    plt.tight_layout()
    plt.savefig(out_dir / "category_nfe_wer_heatmap.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # 5) Category-wise NFE curves
    # ------------------------------------------------------------
    plt.figure(figsize=(9, 6))

    for category in sorted(cat_nfe["category"].unique()):
        subset = cat_nfe[cat_nfe["category"] == category].sort_values("nfe")
        plt.plot(
            subset["nfe"],
            subset["mean_wer"],
            marker="o",
            label=category
        )

    plt.xlabel("NFE / sampling steps")
    plt.ylabel("Mean WER")
    plt.title("NFE Sensitivity by Input Category")
    plt.xticks(NFE_ORDER)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "category_nfe_wer_curves.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # 6) Select the most NFE-sensitive sentences
    # ------------------------------------------------------------
    nfe_cols = [str(n) for n in NFE_ORDER if str(n) in cases.columns]

    if not nfe_cols:
        # Sometimes pandas may preserve numeric-looking column names differently.
        nfe_cols = [c for c in cases.columns if str(c) in {"1", "2", "4", "8", "16"}]

    if not nfe_cols:
        raise RuntimeError(
            "Could not find NFE columns (1,2,4,8,16) in case_study_candidates.csv"
        )

    for c in nfe_cols:
        cases[c] = pd.to_numeric(cases[c], errors="coerce")

    cases["wer_range"] = cases[nfe_cols].max(axis=1) - cases[nfe_cols].min(axis=1)

    top_cases = (
        cases.sort_values("wer_range", ascending=False)
        .head(args.top_cases)
        .copy()
    )

    top_cases.to_csv(
        out_dir / "selected_case_studies.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # 7) One plot per selected case
    # ------------------------------------------------------------
    case_lines = []

    for rank, (_, row) in enumerate(top_cases.iterrows(), start=1):
        x = []
        y = []

        for nfe in NFE_ORDER:
            col = str(nfe)
            if col in row.index and pd.notna(row[col]):
                x.append(nfe)
                y.append(float(row[col]))

        title_text = str(row["text"])
        short_title = title_text if len(title_text) <= 65 else title_text[:62] + "..."

        save_line_plot(
            x,
            y,
            "NFE / sampling steps",
            "WER",
            f"Case {rank}: {short_title}",
            case_dir / f"case_{rank:02d}.png",
            xticks=NFE_ORDER,
        )

        case_lines.append(
            {
                "rank": rank,
                "category": row["category"],
                "item_id": int(row["item_id"]),
                "text": row["text"],
                "wer_range": row["wer_range"],
                **{
                    f"wer_nfe_{nfe}": row[str(nfe)]
                    if str(nfe) in row.index
                    else np.nan
                    for nfe in NFE_ORDER
                }
            }
        )

    pd.DataFrame(case_lines).to_csv(
        out_dir / "selected_case_studies_clean.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # 8) Auto-generate a short findings note
    # ------------------------------------------------------------
    best_speed_row = by_nfe.loc[by_nfe["mean_rtf"].idxmin()]
    lowest_wer_row = by_nfe.loc[by_nfe["mean_wer"].idxmin()]
    hardest_category = by_category.loc[by_category["mean_wer"].idxmax()]
    easiest_category = by_category.loc[by_category["mean_wer"].idxmin()]

    report = []
    report.append("# Day 10 Findings\n")
    report.append("## Overall observations\n")
    report.append(
        f"- Fastest average setting: NFE {int(best_speed_row['nfe'])} "
        f"(mean RTF = {best_speed_row['mean_rtf']:.4f})."
    )
    report.append(
        f"- Lowest average WER: NFE {int(lowest_wer_row['nfe'])} "
        f"(mean WER = {lowest_wer_row['mean_wer']:.4f})."
    )
    report.append(
        f"- Highest-WER category overall: {hardest_category['category']} "
        f"(mean WER = {hardest_category['mean_wer']:.4f})."
    )
    report.append(
        f"- Lowest-WER category overall: {easiest_category['category']} "
        f"(mean WER = {easiest_category['mean_wer']:.4f})."
    )

    report.append("\n## Selected NFE-sensitive cases\n")
    for item in case_lines:
        report.append(
            f"- Case {item['rank']} | {item['category']} | "
            f"WER range = {item['wer_range']:.3f} | {item['text']}"
        )

    report.append("\n## Interpretation caution\n")
    report.append(
        "- WER is measured by Whisper, so a WER difference does not automatically "
        "mean the TTS audio itself is perceptually better or worse."
    )
    report.append(
        "- Selected cases should be checked by listening before making a strong "
        "claim about TTS robustness."
    )

    (out_dir / "DAY10_FINDINGS.md").write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\nDay 10 visualization complete.")
    print(f"Output folder: {out_dir}")
    print("Main outputs:")
    print(" - overall_nfe_vs_wer.png")
    print(" - overall_nfe_vs_rtf.png")
    print(" - mean_wer_by_category.png")
    print(" - category_nfe_wer_heatmap.png")
    print(" - category_nfe_wer_curves.png")
    print(" - selected_case_studies.csv")
    print(" - case_studies/case_01.png ...")
    print(" - DAY10_FINDINGS.md")


if __name__ == "__main__":
    main()
