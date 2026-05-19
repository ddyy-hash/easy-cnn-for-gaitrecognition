"""Train the easy 3D CNN gait recognition baseline."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, to_categorical

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from easy_gait_cnn.model import build_3d_cnn  # noqa: E402
from easy_gait_cnn.video import load_video_dataset  # noqa: E402


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def plot_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    accuracy = history.history.get("accuracy", history.history.get("acc"))
    val_accuracy = history.history.get("val_accuracy", history.history.get("val_acc"))

    plt.figure(figsize=(7, 4))
    plt.plot(accuracy, marker=".", label="train")
    plt.plot(val_accuracy, marker=".", label="validation")
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(history.history["loss"], marker=".", label="train")
    plt.plot(history.history["val_loss"], marker=".", label="validation")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_loss.png", dpi=160)
    plt.close()


def save_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    accuracy = history.history.get("accuracy", history.history.get("acc"))
    val_accuracy = history.history.get("val_accuracy", history.history.get("val_acc"))

    with (output_dir / "result.tsv").open("w", encoding="utf-8") as handle:
        handle.write("epoch\tloss\taccuracy\tval_loss\tval_accuracy\n")
        for epoch, (loss, acc, val_loss, val_acc) in enumerate(
            zip(history.history["loss"], accuracy, history.history["val_loss"], val_accuracy, strict=True)
        ):
            handle.write(f"{epoch}\t{loss}\t{acc}\t{val_loss}\t{val_acc}\n")


def load_or_build_dataset(args: argparse.Namespace, output_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    channels = 3 if args.color else 1
    cache_path = output_dir / f"dataset_{args.class_limit}_{args.skip}_{args.depth}_{args.height}_{channels}.npz"

    if args.cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return cached["x"], cached["y"], cached["classes"].tolist()

    x, y, classes = load_video_dataset(
        args.videos,
        class_limit=args.class_limit,
        color=args.color,
        skip=args.skip,
        width=args.width,
        height=args.height,
        depth=args.depth,
    )

    if args.cache:
        np.savez_compressed(cache_path, x=x, y=y, classes=np.asarray(classes))

    return x, y, classes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a compact 3D CNN for gait recognition.")
    parser.add_argument("--videos", type=Path, default=Path("data/private"), help="Directory with private labeled videos.")
    parser.add_argument("--output", type=Path, default=Path("outputs/3dcnn_6class"), help="Output directory.")
    parser.add_argument("--class-limit", type=int, default=6, help="Maximum number of classes to load.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--width", type=int, default=64, help="Frame width.")
    parser.add_argument("--height", type=int, default=64, help="Frame height.")
    parser.add_argument("--depth", type=int, default=16, help="Number of sampled frames per clip.")
    parser.add_argument("--color", type=str_to_bool, default=True, help="Use RGB frames.")
    parser.add_argument("--skip", type=str_to_bool, default=True, help="Sample frames across the full clip.")
    parser.add_argument("--cache", type=str_to_bool, default=True, help="Cache processed tensors in the output folder.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(args.seed)

    x, labels, classes = load_or_build_dataset(args, output_dir)
    y = to_categorical(labels, num_classes=len(classes))
    (output_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    label_counts = Counter(labels.tolist())
    stratify = labels if min(label_counts.values()) >= 2 and len(labels) * args.test_size >= len(classes) else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
        stratify=stratify,
    )

    model = build_3d_cnn(input_shape=x.shape[1:], num_classes=len(classes))
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=["accuracy"])
    model.summary()

    try:
        plot_model(model, show_shapes=True, to_file=output_dir / "model.png")
    except Exception as exc:  # Graphviz is optional for training.
        print(f"Model plot skipped: {exc}")

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        shuffle=True,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Validation loss: {loss:.6f}")
    print(f"Validation accuracy: {accuracy:.6f}")

    (output_dir / "easy_gait_3dcnn.json").write_text(model.to_json(), encoding="utf-8")
    model.save_weights(output_dir / "easy_gait_3dcnn.weights.h5")
    plot_history(history, output_dir)
    save_history(history, output_dir)


if __name__ == "__main__":
    main()
