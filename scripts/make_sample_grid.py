"""Create a visual grid of representative gait frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from easy_gait_cnn.video import list_videos, parse_label  # noqa: E402


def read_preview_frame(path: Path, width: int = 240, height: int = 180):
    """Read one still frame for public documentation screenshots."""
    frame = iio.imread(path, index=0)
    image = Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a gait frame grid from private videos for README documentation.")
    parser.add_argument("--videos", type=Path, default=Path("data/private"), help="Directory with private videos.")
    parser.add_argument("--output", type=Path, default=Path("docs/figures/sample_frame_grid.png"), help="Output image.")
    parser.add_argument("--limit", type=int, default=6, help="Maximum number of labels to show.")
    args = parser.parse_args()

    videos = list_videos(args.videos)
    selected: dict[str, Path] = {}
    for video in videos:
        label = parse_label(video)
        selected.setdefault(label, video)
        if len(selected) >= args.limit:
            break

    if not selected:
        raise FileNotFoundError(f"No videos found in: {args.videos}")

    columns = min(3, len(selected))
    rows = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 2.7))
    axes = [axes] if len(selected) == 1 else axes.ravel()

    for axis, (label, video) in zip(axes, selected.items(), strict=False):
        axis.imshow(read_preview_frame(video))
        axis.set_title(f"Class {label}", fontsize=11)
        axis.axis("off")

    for axis in axes[len(selected) :]:
        axis.axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
