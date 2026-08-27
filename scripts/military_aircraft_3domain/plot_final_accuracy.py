import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "exp" / "military_aircraft_3domain" / "final_results"


def load_curve(filename):
    with (RESULT_ROOT / "accuracy" / filename).open(
            "r", encoding="utf-8") as stream:
        rows = json.load(stream)["rounds"]
    return [row["round"] for row in rows], [row["average"] for row in rows]


def main():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    curves = [
        ("FedAvg", "fedavg_accuracy_100round.json", "#2563EB", "--", "o"),
        ("FedProx", "fedprox_accuracy_100round.json", "#EA580C", "-.", "s"),
        ("平台", "platform_accuracy_100round.json", "#16A34A", "-", "D"),
    ]

    fig, axis = plt.subplots(figsize=(11.2, 6.4), facecolor="white")
    axis.set_facecolor("white")
    for label, filename, color, linestyle, marker in curves:
        rounds, accuracy = load_curve(filename)
        axis.plot(
            rounds,
            accuracy,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.4 if label == "平台" else 2.0,
            marker=marker,
            markevery=10,
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )
        if label == "FedProx":
            label_offset = (-8, -14)
            vertical_alignment = "top"
        else:
            label_offset = (-8, 9)
            vertical_alignment = "bottom"
        axis.annotate(
            f"{accuracy[-1] * 100:.2f}%",
            xy=(rounds[-1], accuracy[-1]),
            xytext=label_offset,
            textcoords="offset points",
            ha="right",
            va=vertical_alignment,
            color=color,
            fontsize=10.5,
            fontweight="medium",
        )

    axis.set_title("MilitaryAircraft3D：ViT+MLP 模型准确率曲线", fontsize=16, pad=14)
    axis.set_xlabel("联邦训练轮次", fontsize=12)
    axis.set_ylabel("平均模型准确率", fontsize=12)
    axis.set_xlim(0, 99)
    axis.set_ylim(0.15, 0.86)
    axis.set_xticks(list(range(0, 100, 10)) + [99])
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.grid(True, which="major", color="#CBD5E1", linewidth=0.8, alpha=0.65)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#64748B")
        spine.set_linewidth(0.9)
    axis.legend(loc="lower right", frameon=True, fontsize=11, ncol=1)
    fig.tight_layout()

    png_path = RESULT_ROOT / "military_accuracy_curves.png"
    svg_path = RESULT_ROOT / "military_accuracy_curves.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"PNG={png_path}")
    print(f"SVG={svg_path}")


if __name__ == "__main__":
    main()
