"""Video loading utilities for fixed-length 3D CNN input tensors."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def parse_label(path: str | Path) -> str:
    """Parse the class label from a file name such as ``001-01-1.mp4``."""
    stem = Path(path).stem
    label = stem.split("-", maxsplit=1)[0].strip()
    if not label:
        raise ValueError(f"Cannot parse a label from file name: {path}")
    return label


def list_videos(video_dir: str | Path) -> list[Path]:
    """Return video files sorted by name."""
    root = Path(video_dir)
    if not root.exists():
        raise FileNotFoundError(f"Video directory does not exist: {root}")
    return sorted(path for path in root.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)


class VideoToTensor:
    """Convert one video into a normalized fixed-depth tensor."""

    def __init__(self, width: int = 64, height: int = 64, depth: int = 16) -> None:
        self.width = width
        self.height = height
        self.depth = depth

    def read(self, filename: str | Path, *, color: bool = True, skip: bool = True) -> np.ndarray:
        """Read a video as ``(frames, height, width, channels)``.

        The converter samples a fixed number of frames. If a clip is shorter
        than the requested depth, the previous valid frame is repeated.
        """
        import cv2

        cap = cv2.VideoCapture(str(filename))
        if not cap.isOpened():
            raise OSError(f"OpenCV could not open video: {filename}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if skip and frame_count > 1:
            frame_indices = np.linspace(0, frame_count - 1, self.depth, dtype=int)
        else:
            frame_indices = np.arange(self.depth, dtype=int)

        frames = []
        last_frame: np.ndarray | None = None

        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = self._fallback_frame(last_frame)
            else:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                last_frame = frame

            if color:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[..., np.newaxis]

            frames.append(frame)

        cap.release()
        return np.asarray(frames, dtype=np.float32) / 255.0

    def _fallback_frame(self, last_frame: np.ndarray | None) -> np.ndarray:
        if last_frame is not None:
            return last_frame.copy()
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


def load_video_dataset(
    video_dir: str | Path,
    *,
    class_limit: int | None = None,
    color: bool = True,
    skip: bool = True,
    width: int = 64,
    height: int = 64,
    depth: int = 16,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a directory of labeled videos into tensors and integer labels."""
    files = list_videos(video_dir)
    if not files:
        raise FileNotFoundError(f"No supported videos found in: {video_dir}")

    labels = [parse_label(path) for path in files]
    classes = sorted(dict.fromkeys(labels))
    if class_limit is not None:
        classes = classes[:class_limit]

    class_to_index = {label: index for index, label in enumerate(classes)}
    selected = [(path, label) for path, label in zip(files, labels, strict=True) if label in class_to_index]
    if not selected:
        raise ValueError(f"No videos matched the selected classes: {classes}")

    converter = VideoToTensor(width=width, height=height, depth=depth)
    x = np.stack([converter.read(path, color=color, skip=skip) for path, _ in selected], axis=0)
    y = np.asarray([class_to_index[label] for _, label in selected], dtype=np.int64)

    counts = Counter(y.tolist())
    missing = [label for label, index in class_to_index.items() if counts.get(index, 0) == 0]
    if missing:
        raise ValueError(f"Selected classes have no videos: {missing}")

    return x, y, classes
